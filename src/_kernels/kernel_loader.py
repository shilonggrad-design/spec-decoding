"""
kernel_loader.py — Python wrapper for GrammarSD CUDA kernels.

Tries to import the compiled CUDA extension. Falls back to Python reference
implementations if kernels are not built (e.g., during Week 2 Days 1-3
before CUDA setup, or on CPU-only machines).

Usage:
    from src._kernels.kernel_loader import popcount_density, grammar_masked_argmax, fused_sample

    # Returns (valid_count, density) for a bitmask
    count, density = popcount_density(bitmask, vocab_size)

    # Returns argmax over masked logits
    best_idx = grammar_masked_argmax(logits, bitmask, vocab_size)

    # Returns sampled token (greedy by default)
    token = fused_sample(logits, bitmask, vocab_size, temperature=0)
"""

from __future__ import annotations

import math

import torch

# ---------------------------------------------------------------------------
# Try to load compiled CUDA extension
# ---------------------------------------------------------------------------
_CUDA_AVAILABLE = False
try:
    import grammar_sd_kernels
    _CUDA_AVAILABLE = True
except ImportError:
    grammar_sd_kernels = None  # type: ignore


def is_cuda_available() -> bool:
    """Return True if compiled CUDA kernels are available."""
    return _CUDA_AVAILABLE


# ---------------------------------------------------------------------------
# Python reference implementations (fallback)
# ---------------------------------------------------------------------------

def _python_popcount(bitmask_row: torch.Tensor, vocab_size: int) -> tuple[int, float]:
    """Count valid tokens in packed int32 bitmask (Python reference)."""
    valid = 0
    for word in bitmask_row:
        bits = word.item()
        if bits < 0:
            bits += 1 << 32
        valid += bin(bits).count("1")
    return valid, valid / vocab_size


def _python_masked_argmax(logits: torch.Tensor, bitmask_row: torch.Tensor, vocab_size: int) -> int:
    """Apply grammar mask to logits and return argmax (Python reference)."""
    masked = logits.clone()
    for i in range(vocab_size):
        word_idx = i // 32
        bit_idx = i % 32
        valid = (bitmask_row[word_idx].item() >> bit_idx) & 1 if word_idx < len(bitmask_row) else 0
        if not valid:
            masked[i] = float("-inf")
    return masked.argmax().item()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def popcount_density(
    bitmask: torch.Tensor,
    vocab_size: int,
    device: str | torch.device = "cuda",
) -> tuple[int, float]:
    """Count valid tokens in grammar bitmask and return density.

    Args:
        bitmask: packed int32 tensor, shape [num_words] or [batch, num_words]
        vocab_size: total vocabulary size
        device: target device

    Returns:
        (valid_count, density) where density = valid_count / vocab_size
    """
    # Handle batched bitmask [1, num_words] → [num_words]
    if bitmask.dim() > 1:
        bitmask_row = bitmask[0]
    else:
        bitmask_row = bitmask

    num_words = bitmask_row.shape[0]

    if _CUDA_AVAILABLE and torch.cuda.is_available():
        bitmask_gpu = bitmask_row.to(device).to(torch.int32)
        total = torch.zeros(1, dtype=torch.int32, device=device)
        grammar_sd_kernels.popcount_density(bitmask_gpu, total, num_words)
        count = total.item()
        return count, count / vocab_size
    else:
        return _python_popcount(bitmask_row, vocab_size)


def grammar_masked_argmax(
    logits: torch.Tensor,      # [vocab_size] or [K+1, vocab_size]
    bitmask: torch.Tensor,     # [num_words] or [K+1, num_words]
    vocab_size: int,
    num_words: int,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    """Fused grammar mask + argmax.

    Args:
        logits: [vocab_size] for single position, or [K+1, vocab_size] for batch
        bitmask: [num_words] for single position, or [K+1, num_words] for batch
        vocab_size: total vocabulary size
        num_words: bitmask length = ceil(vocab_size / 32)
        device: target device

    Returns:
        int (single position) or Tensor[K+1] (batch)
    """
    # Batch mode: [K+1, vocab_size]
    if logits.dim() == 2:
        K_plus_1 = logits.size(0)
        result = torch.zeros(K_plus_1, dtype=torch.int32, device=device)

        if _CUDA_AVAILABLE and torch.cuda.is_available():
            grammar_sd_kernels.grammar_masked_argmax(
                logits.to(device).to(torch.float32),
                bitmask.to(device).to(torch.int32),
                result,
                vocab_size,
                num_words,
            )
            return result
        else:
            for pos in range(K_plus_1):
                result[pos] = _python_masked_argmax(logits[pos].cpu(), bitmask[pos], vocab_size)
            return result

    # Single position mode: [vocab_size]
    else:
        if _CUDA_AVAILABLE and torch.cuda.is_available():
            # Reshape to [1, vocab_size] and [1, num_words] for batch kernel
            logits_2d = logits.unsqueeze(0).to(device).to(torch.float32)
            bitmask_2d = bitmask.unsqueeze(0).to(device).to(torch.int32)
            result = torch.zeros(1, dtype=torch.int32, device=device)
            grammar_sd_kernels.grammar_masked_argmax(
                logits_2d, bitmask_2d, result, vocab_size, num_words
            )
            return result[0]
        else:
            return torch.tensor(
                _python_masked_argmax(logits.cpu(), bitmask.cpu(), vocab_size),
                dtype=torch.int32,
            )


def fused_sample(
    logits: torch.Tensor,      # [vocab_size]
    bitmask: torch.Tensor,     # [num_words]
    vocab_size: int,
    num_words: int,
    temperature: float = 0.0,
    seed: int = 42,
    device: str | torch.device = "cuda",
) -> int:
    """Fused grammar mask + softmax + sample for single position.

    Args:
        logits: [vocab_size] float
        bitmask: [num_words] int32
        vocab_size: total vocabulary size
        num_words: bitmask length
        temperature: 0 for greedy (argmax), >0 for sampling
        seed: random seed for sampling
        device: target device

    Returns:
        sampled token id (int)
    """
    if _CUDA_AVAILABLE and torch.cuda.is_available():
        logits_gpu = logits.to(device).to(torch.float32)
        bitmask_gpu = bitmask.to(device).to(torch.int32)
        sampled = torch.zeros(1, dtype=torch.int32, device=device)
        grammar_sd_kernels.fused_sample(
            logits_gpu, bitmask_gpu, sampled,
            temperature, vocab_size, num_words, seed
        )
        return sampled.item()
    else:
        # Python fallback: mask + argmax (greedy only for now)
        return _python_masked_argmax(logits.cpu(), bitmask.cpu(), vocab_size)
