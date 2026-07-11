"""
Triton fused logit processor — grammar-masked argmax + density in a single GPU pass.

Why this kernel exists
=====================
The verify/accept-reject loop in VeloSpec currently does 3 separate operations
per token position:

  1. xgrammar.apply_token_bitmask_inplace(logits, bitmask)  — kernel launch #1
  2. logits.argmax()                                        — kernel launch #2
  3. Python popcount loop for density                       — CPU, O(n)

This kernel fuses all three into ONE kernel launch:

  ┌──────────────────────────────────────────────────────┐
  │  load logits    │  load bitmask words                   │
  │  extract bits   │  mask illegal tokens → -inf          │
  │  find argmax    │  count valid tokens (density)         │
  └──────────────────────────────────────────────────────┘

Savings per verify round (K+1 positions):
  - Eliminates 2×(K+1) kernel launches → 2 kernel launches (2-pass reduction)
  - Eliminates Python popcount loop entirely

Architecture
============
Two-pass reduction (vocab_size = 248320 > max Triton block size):

  Pass 1 (N blocks): Each block finds local argmax (value + index) + valid count.
  Pass 2 (1 block):  Finds global argmax across all blocks + sums valid counts.

Each token loads its own bitmask word directly: word_idx = token_idx // 32.
Adjacent tokens (within same int32 word) hit the same cache line → efficient.

Fallback: If Triton is not available, engine.py uses xgrammar + PyTorch directly.
"""

from __future__ import annotations

import torch

_TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]


def is_available() -> bool:
    """Check if Triton GPU backend is available."""
    return _TRITON_AVAILABLE and torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Pass 1: Block-level reduction
# ---------------------------------------------------------------------------
if _TRITON_AVAILABLE:
    @triton.jit
    def _block_reduce_kernel(
        # Inputs
        logits_ptr,       # float32*  — [vocab_size]
        bitmask_ptr,      # int32*    — [num_words]
        # Outputs (one element per block)
        block_max_val,    # float32*  — max logit value in this block
        block_max_idx,    # int32*    — token index of that max
        block_valid_cnt,  # int32*    — number of valid (bit=1) tokens
        # Constants
        vocab_size: tl.int32,
        num_words: tl.int32,
        BLOCK_SIZE: tl.constexpr,       # tokens per block (e.g. 4096)
    ):
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        in_range = offsets < vocab_size

        # --- Load logits ---
        logits = tl.load(logits_ptr + offsets, mask=in_range, other=float('-inf'))

        # --- Load bitmask: each token loads its own word directly ---
        word_offsets = offsets // 32
        word_in_range = word_offsets < num_words
        words = tl.load(bitmask_ptr + word_offsets, mask=word_in_range, other=0)

        # Extract bit for each token: bit = (word >> bit_pos) & 1
        bit_pos = offsets % 32
        bits = (words >> bit_pos) & 1

        # --- Apply grammar mask: illegal tokens → -inf ---
        masked_logits = tl.where(bits == 1, logits, float('-inf'))

        # --- Block reductions ---

        # 1) Max logit value in this block
        block_max = tl.max(masked_logits, axis=0)

        # 2) Argmax index: among positions that have block_max AND bit=1
        is_max = (masked_logits == block_max) & (bits == 1) & in_range
        idx_candidates = tl.where(is_max, offsets, vocab_size)
        block_argmax = tl.min(idx_candidates, axis=0)

        # 3) Valid token count
        valid_count = tl.sum(
            (bits.to(tl.int32) & in_range.to(tl.int32)), axis=0
        )

        # --- Store block results ---
        tl.store(block_max_val + pid, block_max)
        tl.store(block_max_idx + pid, block_argmax)
        tl.store(block_valid_cnt + pid, valid_count)


# ---------------------------------------------------------------------------
# Pass 2: Cross-block final reduction
# ---------------------------------------------------------------------------
if _TRITON_AVAILABLE:
    @triton.jit
    def _final_reduce_kernel(
        block_max_val,    # float32* [num_blocks]
        block_max_idx,    # int32*   [num_blocks]
        block_valid_cnt,  # int32*   [num_blocks]
        out_argmax,       # int32*   [1]
        out_valid_total,  # float32* [1]
        num_blocks: tl.int32,
        vocab_size: tl.int32,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.arange(0, BLOCK_SIZE)
        in_range = offsets < num_blocks

        max_vals = tl.load(block_max_val + offsets, mask=in_range, other=float('-inf'))
        max_idxs = tl.load(block_max_idx + offsets, mask=in_range, other=vocab_size)
        valid_cnts = tl.load(
            block_valid_cnt + offsets, mask=in_range, other=tl.int32(0)
        )

        # Global max value
        global_max = tl.max(max_vals, axis=0)

        # Find the index that holds the global max
        is_global_max = (max_vals == global_max) & in_range
        idx_candidates = tl.where(is_global_max, max_idxs, vocab_size)
        final_argmax = tl.min(idx_candidates, axis=0)

        # Total valid tokens
        total_valid = tl.sum(valid_cnts, axis=0)

        tl.store(out_argmax, final_argmax)
        tl.store(out_valid_total, total_valid.to(tl.float32))


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------
DEFAULT_BLOCK_SIZE = 4096  # tokens per block


def fused_masked_argmax(
    logits: torch.Tensor,
    bitmask: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> tuple[int, float]:
    """Fused grammar-masked argmax + density — single GPU kernel pipeline.

    Args:
        logits: Shape ``[vocab_size]``, float16 or float32, on GPU.
        bitmask: Shape ``[num_words]``, int32, on GPU (xgrammar bitmask).
        block_size: Tokens per Triton block (default 4096).

    Returns:
        ``(token_id, density)`` where *density* = valid_tokens / vocab_size.

    Eliminates 2 kernel launches and 1 CPU popcount loop per call.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError(
            "Triton not installed. Install with: pip install triton"
        )
    if not logits.is_cuda or not bitmask.is_cuda:
        raise ValueError("Both logits and bitmask must be on GPU")

    vocab_size = logits.shape[0]
    num_words = (vocab_size + 31) // 32
    num_blocks = triton.cdiv(vocab_size, block_size)

    # Ensure contiguous; cast to float32 for numerical stability
    logits = logits.contiguous().float()
    bitmask = bitmask.contiguous()

    # --- Allocate intermediate buffers (one element per block) ---
    block_max_val = torch.empty(num_blocks, dtype=torch.float32, device=logits.device)
    block_max_idx = torch.empty(num_blocks, dtype=torch.int32, device=logits.device)
    block_valid_cnt = torch.empty(num_blocks, dtype=torch.int32, device=logits.device)

    # --- Pass 1: block-level reduction ---
    _block_reduce_kernel[(num_blocks,)](
        logits, bitmask,
        block_max_val, block_max_idx, block_valid_cnt,
        vocab_size, num_words,
        BLOCK_SIZE=block_size,
    )

    # --- Pass 2: cross-block final reduction ---
    final_argmax = torch.empty(1, dtype=torch.int32, device=logits.device)
    final_valid = torch.empty(1, dtype=torch.float32, device=logits.device)

    _final_reduce_kernel[(1,)](
        block_max_val, block_max_idx, block_valid_cnt,
        final_argmax, final_valid,
        num_blocks, vocab_size,
        BLOCK_SIZE=min(num_blocks + 1, 1024),
    )

    token_id = final_argmax.item()
    valid_count = final_valid.item()
    density = valid_count / vocab_size if vocab_size > 0 else 0.0

    return token_id, density
