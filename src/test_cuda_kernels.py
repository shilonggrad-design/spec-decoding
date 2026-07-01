#!/usr/bin/env python3
"""
test_cuda_kernels.py — Validate CUDA kernels against Python reference implementations.

Run AFTER building the CUDA extension:
    !python setup.py build_ext --inplace
    !python src/test_cuda_kernels.py

Each test:
  1. Creates a known input (bitmask + logits)
  2. Computes the expected result in Python
  3. Runs the CUDA kernel
  4. Asserts they match

If CUDA extension is not built, tests run Python-only (smoke test).
"""

from __future__ import annotations

import os
import sys
import time

import torch

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src._kernels.kernel_loader import (  # noqa: E402
    is_cuda_available,
    popcount_density,
    grammar_masked_argmax,
    fused_sample,
)

# ---------------------------------------------------------------------------
# Constants matching Qwen3.5
# ---------------------------------------------------------------------------
VOCAB_SIZE = 248_320
NUM_WORDS = (VOCAB_SIZE + 31) // 32  # = 7760


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------
def make_bitmask(valid_indices: list[int]) -> torch.Tensor:
    """Create a packed int32 bitmask with specified token indices set to valid."""
    bitmask = torch.zeros(NUM_WORDS, dtype=torch.int32)
    for idx in valid_indices:
        word_idx = idx // 32
        bit_idx = idx % 32
        # Python int, then cast
        val = bitmask[word_idx].item()
        if val < 0:
            val += 1 << 32
        val |= (1 << bit_idx)
        if val >= (1 << 31):
            val -= (1 << 32)
        bitmask[word_idx] = val
    return bitmask


def python_popcount(bitmask: torch.Tensor, vocab_size: int) -> int:
    """Reference: count valid tokens in bitmask."""
    valid = 0
    for word in bitmask:
        bits = word.item()
        if bits < 0:
            bits += 1 << 32
        valid += bin(bits).count("1")
    return valid


def python_masked_argmax(logits: torch.Tensor, bitmask: torch.Tensor, vocab_size: int) -> int:
    """Reference: apply mask + argmax."""
    masked = logits.clone()
    for i in range(vocab_size):
        word_idx = i // 32
        bit_idx = i % 32
        word_val = bitmask[word_idx].item()
        if word_val < 0:
            word_val += 1 << 32
        valid = (word_val >> bit_idx) & 1
        if not valid:
            masked[i] = float("-inf")
    return masked.argmax().item()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_popcount_basic():
    """Kernel 1: popcount matches Python for simple bitmask."""
    # Valid: tokens 0-47 (2 full words)
    valid_indices = list(range(48))
    bitmask = make_bitmask(valid_indices)

    py_count = python_popcount(bitmask, VOCAB_SIZE)

    if is_cuda_available() and torch.cuda.is_available():
        count, density = popcount_density(bitmask, VOCAB_SIZE)
        assert count == py_count, f"Popcount mismatch: CUDA={count}, Python={py_count}"
        assert abs(density - py_count / VOCAB_SIZE) < 1e-10, "Density mismatch"
        print(f"  ✅ Kernel 1 (popcount): CUDA={count} == Python={py_count}")
    else:
        print(f"  ⚠️  Kernel 1 skipped (no CUDA), Python={py_count}")


def test_popcount_sparse():
    """Kernel 1: popcount for sparse bitmask (3 valid tokens)."""
    valid_indices = [100, 5000, 100000]
    bitmask = make_bitmask(valid_indices)

    py_count = python_popcount(bitmask, VOCAB_SIZE)

    if is_cuda_available() and torch.cuda.is_available():
        count, density = popcount_density(bitmask, VOCAB_SIZE)
        assert count == py_count, f"Popcount mismatch: CUDA={count}, Python={py_count}"
        print(f"  ✅ Kernel 1 (sparse): CUDA={count} == Python={py_count}")
    else:
        print(f"  ⚠️  Kernel 1 sparse skipped (no CUDA), Python={py_count}")


def test_argmax_basic():
    """Kernel 2: grammar_masked_argmax matches Python."""
    # Create logits where token 100 has highest value, but token 0 has higher value (invalid)
    logits = torch.randn(VOCAB_SIZE) * 0.1
    logits[0] = 999.0    # Would be argmax but invalid
    logits[100] = 42.0   # Should be argmax after masking
    logits[5000] = 41.0  # Valid but lower

    bitmask = make_bitmask([100, 5000])

    py_result = python_masked_argmax(logits, bitmask, VOCAB_SIZE)

    if is_cuda_available() and torch.cuda.is_available():
        cu_result = grammar_masked_argmax(
            logits.cuda(), bitmask.cuda(), VOCAB_SIZE, NUM_WORDS
        )
        cu_result = int(cu_result.item()) if cu_result.dim() == 0 else int(cu_result[0].item())
        assert cu_result == py_result, \
            f"Argmax mismatch: CUDA={cu_result}, Python={py_result}"
        print(f"  ✅ Kernel 2 (argmax): CUDA={cu_result} == Python={py_result}")
    else:
        print(f"  ⚠️  Kernel 2 skipped (no CUDA), Python={py_result}")


def test_fused_sample_greedy():
    """Kernel 3: fused_sample greedy (temperature=0) matches argmax."""
    logits = torch.randn(VOCAB_SIZE) * 0.1
    logits[50] = 100.0   # Clear argmax
    logits[51] = 99.0    # Close but lower

    valid_indices = [50, 51, 52, 53]
    bitmask = make_bitmask(valid_indices)

    py_result = python_masked_argmax(logits, bitmask, VOCAB_SIZE)

    if is_cuda_available() and torch.cuda.is_available():
        cu_result = fused_sample(
            logits.cuda(), bitmask.cuda(),
            VOCAB_SIZE, NUM_WORDS,
            temperature=0.0, seed=42
        )
        assert cu_result == py_result, \
            f"Fused sample mismatch: CUDA={cu_result}, Python={py_result}"
        print(f"  ✅ Kernel 3 (greedy): CUDA={cu_result} == Python={py_result}")
    else:
        print(f"  ⚠️  Kernel 3 skipped (no CUDA), Python={py_result}")


def test_argmax_batch():
    """Kernel 2: grammar_masked_argmax for K+1=6 positions simultaneously.

    This is the real verify-path scenario: the target model produces logits
    for [prefix + K draft tokens] = K+1 positions, each with its own grammar
    bitmask. The kernel launches one block per position.
    """
    K_plus_1 = 6  # 5 draft tokens + 1 bonus

    # Each position has different valid tokens
    all_valid = [
        [100, 200, 300],       # position 0: token 300 has highest logit
        [10, 20, 30],          # position 1: token 30 has highest
        [3, 4, 5],             # position 2: token 5 has highest
        [500, 501, 502, 503],  # position 3: token 503 has highest
        [999, 1000, 1001],     # position 4: token 1001 has highest
        [777, 888],            # position 5: token 888 has highest
    ]

    # Token 0 is NEVER valid in any position — use it as the "high but invalid" decoy
    DECOY_TOKEN = 0

    # Build logits: random noise + peak at the correct token for each position
    logits_batch = torch.randn(K_plus_1, VOCAB_SIZE) * 0.1
    expected = []
    for pos, valid_indices in enumerate(all_valid):
        # Make the last valid token have a very high logit
        peak_token = valid_indices[-1]
        logits_batch[pos, peak_token] = 42.0 + pos
        # Set a higher INVALID logit to ensure masking works
        logits_batch[pos, DECOY_TOKEN] = 999.0  # token 0 not in any valid set
        expected.append(peak_token)

    # Build batched bitmask [K+1, num_words]
    bitmask_batch = torch.zeros(K_plus_1, NUM_WORDS, dtype=torch.int32)
    for pos, valid_indices in enumerate(all_valid):
        row = make_bitmask(valid_indices)
        bitmask_batch[pos] = row

    if is_cuda_available() and torch.cuda.is_available():
        cu_result = grammar_masked_argmax(
            logits_batch.cuda(), bitmask_batch.cuda(),
            VOCAB_SIZE, NUM_WORDS,
        )
        # cu_result is a tensor of shape [K+1]
        cu_indices = cu_result.cpu().tolist()
        for pos in range(K_plus_1):
            assert cu_indices[pos] == expected[pos], \
                f"Position {pos} mismatch: CUDA={cu_indices[pos]}, expected={expected[pos]}"
        print(f"  ✅ Kernel 2 (batch K+1={K_plus_1}): all positions match")
    else:
        print(f"  ⚠️  Kernel 2 batch skipped (no CUDA)")
        for pos in range(K_plus_1):
            py_result = python_masked_argmax(logits_batch[pos], bitmask_batch[pos], VOCAB_SIZE)
            assert py_result == expected[pos], f"Python pos {pos} mismatch"
        print(f"  ✅ Kernel 2 (batch, Python fallback): all positions match")


def test_fused_sample_sampling():
    """Kernel 3: fused_sample with temperature > 0 returns a valid token.

    Unlike greedy (temperature=0), sampling mode uses softmax + CDF.
    We can't predict the exact token, but we can verify it's in the valid set.
    """
    logits = torch.randn(VOCAB_SIZE) * 2.0  # more spread for sampling

    valid_indices = [42, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    bitmask = make_bitmask(valid_indices)
    valid_set = set(valid_indices)

    if is_cuda_available() and torch.cuda.is_available():
        # Run multiple times with different seeds
        sampled_tokens = set()
        for seed in range(10):
            token = fused_sample(
                logits.cuda(), bitmask.cuda(),
                VOCAB_SIZE, NUM_WORDS,
                temperature=1.0, seed=seed,
            )
            assert token in valid_set, \
                f"Sampled token {token} not in valid set {valid_set}"
            sampled_tokens.add(token)

        # With 10 seeds and 10 valid tokens with different probabilities,
        # we should see at least 2 different tokens (confirms it's sampling, not argmax)
        assert len(sampled_tokens) >= 2, \
            f"Expected sampling diversity, got same token {sampled_tokens} every time"
        print(f"  ✅ Kernel 3 (sampling temp=1.0): {len(sampled_tokens)} unique tokens "
              f"across 10 seeds, all valid")
    else:
        print(f"  ⚠️  Kernel 3 sampling skipped (no CUDA)")


def test_cuda_speedup():
    """Measure speedup of CUDA popcount vs Python loop."""
    valid_indices = list(range(1240))  # ~0.5% density
    bitmask = make_bitmask(valid_indices)

    # Python timing
    t0 = time.perf_counter()
    for _ in range(10):
        _ = python_popcount(bitmask, VOCAB_SIZE)
    py_time = (time.perf_counter() - t0) / 10

    if is_cuda_available() and torch.cuda.is_available():
        # CUDA timing (including transfer)
        t0 = time.perf_counter()
        for _ in range(10):
            _ = popcount_density(bitmask, VOCAB_SIZE)
        torch.cuda.synchronize()
        cu_time = (time.perf_counter() - t0) / 10

        speedup = py_time / cu_time
        print(f"  ⚡ Popcount speedup: Python={py_time*1000:.1f}ms → CUDA={cu_time*1000:.3f}ms "
              f"({speedup:.0f}×)")
        assert speedup > 1.0, f"CUDA should be faster (got {speedup}×)"
    else:
        print(f"  ⚠️  Speedup test skipped (no CUDA), Python={py_time*1000:.1f}ms")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("CUDA Kernel Validation Tests")
    print("=" * 60)
    print(f"Vocab size: {VOCAB_SIZE}")
    print(f"Bitmask words: {NUM_WORDS}")
    print(f"CUDA available: {is_cuda_available() and torch.cuda.is_available()}")
    if is_cuda_available() and torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    tests = [
        ("Kernel 1: popcount (basic)", test_popcount_basic),
        ("Kernel 1: popcount (sparse)", test_popcount_sparse),
        ("Kernel 2: grammar_masked_argmax (single)", test_argmax_basic),
        ("Kernel 2: grammar_masked_argmax (batch K+1)", test_argmax_batch),
        ("Kernel 3: fused_sample (greedy)", test_fused_sample_greedy),
        ("Kernel 3: fused_sample (sampling temp=1.0)", test_fused_sample_sampling),
        ("Performance: popcount speedup", test_cuda_speedup),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        print(f"\n--- {name} ---")
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  ❌ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
