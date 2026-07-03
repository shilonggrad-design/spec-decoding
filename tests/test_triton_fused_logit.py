"""
Colab test script for VeloSpec Triton fused logit processor.

Run on Colab with A100 GPU:
  1. !git clone https://github.com/shilonggrad-design/spec-decoding
  2. %cd spec-decoding
  3. !pip install triton torch
  4. python tests/test_triton_fused_logit.py
"""

import torch

# ---- Test 1: Triton kernel correctness against PyTorch reference ----
def test_correctness():
    """Verify Triton argmax matches PyTorch reference on synthetic data."""
    from velospec.triton.fused_logit_processor import fused_masked_argmax, is_available

    assert is_available(), "Triton + CUDA required"

    vocab_size = 248320  # Qwen3.5
    num_words = vocab_size // 32  # 7760

    torch.manual_seed(42)
    logits = torch.randn(vocab_size, dtype=torch.float16, device="cuda")

    # Create a random bitmask (each word has random bits set)
    bitmask = torch.randint(0, 2**31, (num_words,), dtype=torch.int32, device="cuda")

    # Reference: PyTorch
    # Expand bitmask to per-token mask
    token_mask = torch.zeros(vocab_size, dtype=torch.bool, device="cuda")
    for w in range(num_words):
        for b in range(32):
            token_idx = w * 32 + b
            if token_idx < vocab_size:
                token_mask[token_idx] = (bitmask[w].item() >> b) & 1

    masked_logits = logits.float().clone()
    masked_logits[~token_mask] = float("-inf")
    ref_argmax = masked_logits.argmax().item()
    ref_density = token_mask.sum().item() / vocab_size

    # Triton
    tri_argmax, tri_density = fused_masked_argmax(logits, bitmask)

    match = tri_argmax == ref_argmax
    print(f"Test 1 - Correctness:")
    print(f"  PyTorch argmax: {ref_argmax}")
    print(f"  Triton argmax: {tri_argmax}")
    print(f"  Match: {'✅' if match else '❌'}")
    print(f"  PyTorch density: {ref_density:.6f}")
    print(f"  Triton density:  {tri_density:.6f}")
    print(f"  Density match: {'✅' if abs(tri_density - ref_density) < 1e-5 else '❌'}")

    assert match, f"Argmax mismatch: {tri_argmax} != {ref_argmax}"
    assert abs(tri_density - ref_density) < 1e-5, f"Density mismatch"

    return match


# ---- Test 2: Known-answer test ----
def test_known_answer():
    """Test with a deterministic case where we know the answer."""
    from velospec.triton.fused_logit_processor import fused_masked_argmax

    vocab_size = 128  # small for clarity
    num_words = vocab_size // 32  # 4

    logits = torch.zeros(vocab_size, dtype=torch.float32, device="cuda")
    logits[42] = 5.0  # highest logit
    logits[99] = 3.0  # second highest

    # Bitmask: only allow tokens 0-63 (first 2 words all 1s, last 2 all 0s)
    # Word 0 = 0xFFFFFFFF, Word 1 = 0xFFFFFFFF, Word 2 = 0, Word 3 = 0
    bitmask = torch.tensor([0xFFFFFFFF, 0xFFFFFFFF, 0, 0], dtype=torch.int32, device="cuda")

    token_id, density = fused_masked_argmax(logits, bitmask)

    # Token 42 is valid (in first 64) and has highest logit
    # Token 99 is NOT valid (in second 64, masked out)
    print(f"\nTest 2 - Known answer:")
    print(f"  Expected argmax: 42 (valid, logit=5.0)")
    print(f"  Got argmax: {token_id}")
    print(f"  Expected density: {64/128:.4f}")
    print(f"  Got density: {density:.4f}")
    print(f"  {'✅' if token_id == 42 else '❌'} {'✅' if abs(density - 0.5) < 1e-5 else '❌'}")

    assert token_id == 42
    assert abs(density - 0.5) < 1e-5


# ---- Test 3: Performance comparison ----
def test_performance():
    """Compare Triton vs PyTorch + xgrammar overhead (using synthetic mask)."""
    from velospec.triton.fused_logit_processor import fused_masked_argmax

    vocab_size = 248320
    num_words = vocab_size // 32

    torch.manual_seed(0)
    logits = torch.randn(vocab_size, dtype=torch.float16, device="cuda")
    bitmask = torch.randint(0, 2**31, (num_words,), dtype=torch.int32, device="cuda")

    # Warmup
    for _ in range(10):
        fused_masked_argmax(logits, bitmask)

    # Benchmark Triton
    import time
    N = 100
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        fused_masked_argmax(logits, bitmask)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    triton_us = (t1 - t0) / N * 1e6

    # Benchmark PyTorch reference (mask + argmax)
    token_mask = torch.zeros(vocab_size, dtype=torch.bool, device="cuda")
    for w in range(num_words):
        for b in range(32):
            idx = w * 32 + b
            if idx < vocab_size:
                token_mask[idx] = (bitmask[w].item() >> b) & 1

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        masked = logits.float().clone()
        masked[~token_mask] = float("-inf")
        _ = masked.argmax().item()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    pytorch_us = (t1 - t0) / N * 1e6

    print(f"\nTest 3 - Performance ({N} iterations, vocab={vocab_size}):")
    print(f"  Triton:  {triton_us:.1f} μs")
    print(f"  PyTorch: {pytorch_us:.1f} μs")
    print(f"  Speedup: {pytorch_us/triton_us:.1f}×")


# ---- Test 4: Edge cases ----
def test_edge_cases():
    from velospec.triton.fused_logit_processor import fused_masked_argmax

    vocab_size = 64
    num_words = 2

    # Case A: all tokens valid
    logits = torch.randn(vocab_size, dtype=torch.float32, device="cuda")
    bitmask = torch.tensor([0xFFFFFFFF, 0xFFFFFFFF], dtype=torch.int32, device="cuda")
    tid, dens = fused_masked_argmax(logits, bitmask)
    assert dens == 1.0
    assert tid == logits.argmax().item()
    print(f"\nTest 4a - All valid: argmax={tid}, density={dens:.2f} ✅")

    # Case B: only token 0 valid
    bitmask = torch.tensor([1, 0], dtype=torch.int32, device="cuda")
    tid, dens = fused_masked_argmax(logits, bitmask)
    assert tid == 0
    assert abs(dens - 1/64) < 1e-5
    print(f"Test 4b - Single valid: argmax={tid}, density={dens:.4f} ✅")


if __name__ == "__main__":
    print("=" * 60)
    print("VeloSpec Triton Fused Logit Processor — Test Suite")
    print("=" * 60)

    test_correctness()
    test_known_answer()
    test_performance()
    test_edge_cases()

    print("\n" + "=" * 60)
    print("All tests passed ✅")
    print("=" * 60)
