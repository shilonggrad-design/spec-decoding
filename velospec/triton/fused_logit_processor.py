"""
Triton fused logit processor — grammar-masked argmax + density in a single GPU pass.

Why this kernel exists
=====================
The verify/accept-reject loop in VeloSpec does 3 separate operations per token:

  1. xgrammar.apply_token_bitmask_inplace(logits, bitmask)  — kernel launch #1
  2. logits.argmax()                                        — kernel launch #2
  3. Python popcount loop for density                       — CPU, O(n)

This kernel fuses all three into TWO kernel launches (two-pass reduction):
mask + argmax + popcount all in one fused pipeline.

Batch mode
=========
In spec decoding's verify phase, K+1 positions need the same operation.
Instead of K+1 sequential calls, batch mode processes all rows in parallel
by adding a batch dimension to the grid.

  Single-row:  grid = (num_blocks,)         → 1 argmax result
  Batch:       grid = (num_blocks, B)       → B argmax results

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


# ===========================================================================
# Single-row kernels (one vocab row at a time)
# ===========================================================================
if _TRITON_AVAILABLE:
    @triton.jit
    def _block_reduce_kernel(
        logits_ptr,       # float32*  — [vocab_size]
        bitmask_ptr,      # int32*    — [num_words]
        block_max_val,    # float32*  — [num_blocks]
        block_max_idx,    # int32*    — [num_blocks]
        block_valid_cnt,  # int32*    — [num_blocks]
        vocab_size: tl.int32,
        num_words: tl.int32,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        in_range = offsets < vocab_size

        logits = tl.load(logits_ptr + offsets, mask=in_range, other=float('-inf'))

        word_offsets = offsets // 32
        word_in_range = word_offsets < num_words
        words = tl.load(bitmask_ptr + word_offsets, mask=word_in_range, other=0)

        bit_pos = offsets % 32
        bits = (words >> bit_pos) & 1

        masked_logits = tl.where(bits == 1, logits, float('-inf'))

        block_max = tl.max(masked_logits, axis=0)
        is_max = (masked_logits == block_max) & (bits == 1) & in_range
        idx_candidates = tl.where(is_max, offsets, vocab_size)
        block_argmax = tl.min(idx_candidates, axis=0)

        valid_count = tl.sum((bits.to(tl.int32) & in_range.to(tl.int32)), axis=0)

        tl.store(block_max_val + pid, block_max)
        tl.store(block_max_idx + pid, block_argmax)
        tl.store(block_valid_cnt + pid, valid_count)


if _TRITON_AVAILABLE:
    @triton.jit
    def _final_reduce_kernel(
        block_max_val, block_max_idx, block_valid_cnt,
        out_argmax, out_valid_total,
        num_blocks: tl.int32,
        vocab_size: tl.int32,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.arange(0, BLOCK_SIZE)
        in_range = offsets < num_blocks

        max_vals = tl.load(block_max_val + offsets, mask=in_range, other=float('-inf'))
        max_idxs = tl.load(block_max_idx + offsets, mask=in_range, other=vocab_size)
        valid_cnts = tl.load(block_valid_cnt + offsets, mask=in_range, other=0)

        global_max = tl.max(max_vals, axis=0)
        is_global_max = (max_vals == global_max) & in_range
        idx_candidates = tl.where(is_global_max, max_idxs, vocab_size)
        final_argmax = tl.min(idx_candidates, axis=0)

        total_valid = tl.sum(valid_cnts, axis=0)

        tl.store(out_argmax, final_argmax)
        tl.store(out_valid_total, total_valid.to(tl.float32))


# ===========================================================================
# Batch kernels (B rows in parallel)
# ===========================================================================
if _TRITON_AVAILABLE:
    @triton.jit
    def _block_reduce_batch_kernel(
        logits_ptr,       # float32*  — [B, vocab_size]
        bitmask_ptr,      # int32*    — [B, num_words]
        block_max_val,    # float32*  — [B, num_blocks]
        block_max_idx,    # int32*    — [B, num_blocks]
        block_valid_cnt,  # int32*    — [B, num_blocks]
        vocab_size: tl.int32,
        num_words: tl.int32,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)    # block index
        pid_b = tl.program_id(1)  # batch index

        # Compute base offsets for this batch row
        logits_base = pid_b * vocab_size
        bitmask_base = pid_b * num_words
        out_base = pid_b * tl.num_programs(0)  # num_blocks

        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        in_range = offsets < vocab_size

        # Load logits for this batch row
        logits = tl.load(logits_ptr + logits_base + offsets, mask=in_range, other=float('-inf'))

        # Load bitmask for this batch row
        word_offsets = offsets // 32
        word_in_range = word_offsets < num_words
        words = tl.load(bitmask_ptr + bitmask_base + word_offsets, mask=word_in_range, other=0)

        bit_pos = offsets % 32
        bits = (words >> bit_pos) & 1

        masked_logits = tl.where(bits == 1, logits, float('-inf'))

        block_max = tl.max(masked_logits, axis=0)
        is_max = (masked_logits == block_max) & (bits == 1) & in_range
        idx_candidates = tl.where(is_max, offsets, vocab_size)
        block_argmax = tl.min(idx_candidates, axis=0)

        valid_count = tl.sum((bits.to(tl.int32) & in_range.to(tl.int32)), axis=0)

        tl.store(block_max_val + out_base + pid, block_max)
        tl.store(block_max_idx + out_base + pid, block_argmax)
        tl.store(block_valid_cnt + out_base + pid, valid_count)


if _TRITON_AVAILABLE:
    @triton.jit
    def _final_reduce_batch_kernel(
        block_max_val,    # float32* [B, num_blocks]
        block_max_idx,    # int32*   [B, num_blocks]
        block_valid_cnt,  # int32*   [B, num_blocks]
        out_argmax,       # int32*   [B]
        out_valid_total,  # float32* [B]
        num_blocks: tl.int32,
        vocab_size: tl.int32,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid_b = tl.program_id(0)  # batch index

        out_base = pid_b * num_blocks
        offsets = tl.arange(0, BLOCK_SIZE)
        in_range = offsets < num_blocks

        max_vals = tl.load(block_max_val + out_base + offsets, mask=in_range, other=float('-inf'))
        max_idxs = tl.load(block_max_idx + out_base + offsets, mask=in_range, other=vocab_size)
        valid_cnts = tl.load(block_valid_cnt + out_base + offsets, mask=in_range, other=0)

        global_max = tl.max(max_vals, axis=0)
        is_global_max = (max_vals == global_max) & in_range
        idx_candidates = tl.where(is_global_max, max_idxs, vocab_size)
        final_argmax = tl.min(idx_candidates, axis=0)

        total_valid = tl.sum(valid_cnts, axis=0)

        tl.store(out_argmax + pid_b, final_argmax)
        tl.store(out_valid_total + pid_b, total_valid.to(tl.float32))


# ===========================================================================
# Python wrapper
# ===========================================================================
DEFAULT_BLOCK_SIZE = 4096

_buffer_cache: dict = {}


def _get_buffers(vocab_size: int, batch_size: int, device: torch.device, block_size: int):
    """Get or create reusable GPU buffers."""
    key = (vocab_size, batch_size, str(device), block_size)
    if key not in _buffer_cache:
        num_blocks = triton.cdiv(vocab_size, block_size)
        _buffer_cache[key] = {
            "block_max_val": torch.empty(batch_size * num_blocks, dtype=torch.float32, device=device),
            "block_max_idx": torch.empty(batch_size * num_blocks, dtype=torch.int32, device=device),
            "block_valid_cnt": torch.empty(batch_size * num_blocks, dtype=torch.int32, device=device),
            "final_argmax": torch.empty(batch_size, dtype=torch.int32, device=device),
            "final_valid": torch.empty(batch_size, dtype=torch.float32, device=device),
        }
    return _buffer_cache[key]


def fused_masked_argmax(
    logits: torch.Tensor,
    bitmask: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_SIZE,
    _sync: bool = True,
) -> tuple[int, float] | tuple[torch.Tensor, torch.Tensor]:
    """Fused grammar-masked argmax + density — single row, optimized.

    Args:
        logits: [vocab_size], float16/float32, GPU.
        bitmask: [num_words], int32, GPU.
        block_size: tokens per block (default 4096).
        _sync: True → return Python (int, float). False → return GPU tensors.

    Returns:
        (token_id, density) or GPU tensors if _sync=False.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton not installed. pip install triton")

    vocab_size = logits.shape[0]
    num_words = (vocab_size + 31) // 32
    num_blocks = triton.cdiv(vocab_size, block_size)
    device = logits.device

    bufs = _get_buffers(vocab_size, 1, device, block_size)

    if logits.dtype != torch.float32:
        logits = logits.to(torch.float32)

    _block_reduce_kernel[(num_blocks,)](
        logits, bitmask,
        bufs["block_max_val"], bufs["block_max_idx"], bufs["block_valid_cnt"],
        vocab_size, num_words,
        BLOCK_SIZE=block_size,
    )

    final_bs = min(triton.next_power_of_2(num_blocks), 1024)
    _final_reduce_kernel[(1,)](
        bufs["block_max_val"], bufs["block_max_idx"], bufs["block_valid_cnt"],
        bufs["final_argmax"], bufs["final_valid"],
        num_blocks, vocab_size,
        BLOCK_SIZE=final_bs,
    )

    if _sync:
        tid = bufs["final_argmax"].item()
        dens = bufs["final_valid"].item() / vocab_size if vocab_size > 0 else 0.0
        return tid, dens
    return bufs["final_argmax"], bufs["final_valid"]


def fused_masked_argmax_batch(
    logits: torch.Tensor,
    bitmask: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused grammar-masked argmax + density — BATCH mode, all rows in parallel.

    This is the high-performance path for spec decoding's verify phase,
    where K+1 positions need the same grammar-masked argmax operation.

    Args:
        logits: [B, vocab_size], float16/float32, GPU.
        bitmask: [B, num_words], int32, GPU.
        block_size: tokens per block (default 4096).

    Returns:
        (argmax_ids [B] int32 GPU, valid_counts [B] float32 GPU)
        — NO CPU sync, fully async. Caller does .item() / .tolist() when needed.

    Example:
        >>> # K=5 verify phase: 6 positions, vocab=248320
        >>> logits = torch.randn(6, 248320, device="cuda", dtype=torch.float16)
        >>> bitmask = torch.randint(0, 2**31, (6, 7760), device="cuda", dtype=torch.int32)
        >>> ids, valids = fused_masked_argmax_batch(logits, bitmask)
        >>> token_ids = ids.tolist()  # sync here
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton not installed. pip install triton")

    assert logits.dim() == 2, f"Expected 2D [B, vocab], got {logits.dim()}D"
    assert bitmask.dim() == 2, f"Expected 2D [B, num_words], got {bitmask.dim()}D"

    batch_size, vocab_size = logits.shape
    num_words = (vocab_size + 31) // 32
    num_blocks = triton.cdiv(vocab_size, block_size)
    device = logits.device

    bufs = _get_buffers(vocab_size, batch_size, device, block_size)

    if logits.dtype != torch.float32:
        logits = logits.to(torch.float32)

    # Pass 1: grid = (num_blocks, batch_size) — all blocks × all rows in parallel
    _block_reduce_batch_kernel[(num_blocks, batch_size)](
        logits, bitmask,
        bufs["block_max_val"], bufs["block_max_idx"], bufs["block_valid_cnt"],
        vocab_size, num_words,
        BLOCK_SIZE=block_size,
    )

    # Pass 2: grid = (batch_size,) — one program per batch row
    final_bs = min(triton.next_power_of_2(num_blocks), 1024)
    _final_reduce_batch_kernel[(batch_size,)](
        bufs["block_max_val"], bufs["block_max_idx"], bufs["block_valid_cnt"],
        bufs["final_argmax"], bufs["final_valid"],
        num_blocks, vocab_size,
        BLOCK_SIZE=final_bs,
    )

    return bufs["final_argmax"], bufs["final_valid"]
