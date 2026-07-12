"""
Colab validation script for VeloSpec Triton kernel.

Usage in Colab:
  !git clone https://github.com/shilonggrad-design/spec-decoding
  %cd spec-decoding
  !pip install triton
  !python tests/test_triton_colab.py
"""

import os
import sys
import time
import torch

# Ensure the repo root is on sys.path so `import velospec` works without pip install
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 60)
print("VeloSpec Triton Kernel — Colab Validation")
print("=" * 60)
print(f"Python:    {sys.version.split()[0]}")
print(f"PyTorch:   {torch.__version__}")
print(f"CUDA:      {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:       {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"SM count:  {props.multi_processor_count}")
    print(f"Mem:       {props.total_memory / 1e9:.1f} GB")
print("=" * 60)

from velospec.triton.fused_logit_processor import fused_masked_argmax, is_available
assert is_available(), "Triton + CUDA required. Make sure you're on a GPU runtime."


# ===========================================================================
# Helper: vectorized bitmask → per-token boolean mask (replaces slow Python loop)
# ===========================================================================
def expand_bitmask(bitmask: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Expand packed int32 bitmask to per-token bool mask — fully vectorized."""
    num_words = (vocab_size + 31) // 32
    # Ensure unsigned interpretation
    words = bitmask.to(torch.int64) & 0xFFFFFFFF  # treat as unsigned 32-bit
    # For each word, expand 32 bits
    bits = torch.zeros(num_words * 32, dtype=torch.bool, device=bitmask.device)
    for b in range(32):
        bits[b::32] = (words & (1 << b)) != 0
    return bits[:vocab_size]


# ===========================================================================
# Test 1: Small-scale known answer (fast, catches obvious bugs)
# ===========================================================================
def test_known_answer():
    """Deterministic case: we know the exact answer."""
    print("\n▶ Test 1: Known-answer (vocab=128)")

    vocab_size = 128
    logits = torch.zeros(vocab_size, dtype=torch.float32, device="cuda")
    logits[42] = 5.0
    logits[99] = 3.0

    # Allow only first 64 tokens (all bits set = -1 in signed int32)
    bitmask = torch.tensor([-1, -1, 0, 0], dtype=torch.int32, device="cuda")

    token_id, density = fused_masked_argmax(logits, bitmask)

    # Token 42 is valid + highest logit → answer
    # Token 99 is masked out
    assert token_id == 42, f"Expected 42, got {token_id}"
    assert abs(density - 0.5) < 1e-5, f"Expected 0.5, got {density}"

    print(f"  argmax:  {token_id} ✅ (expected 42)")
    print(f"  density: {density:.4f} ✅ (expected 0.5000)")


# ===========================================================================
# Test 2: Correctness on full vocab (Qwen3.5 size)
# ===========================================================================
def test_correctness_full():
    """Triton argmax vs PyTorch reference on vocab_size=248320."""
    print("\n▶ Test 2: Correctness (vocab=248320, Qwen3.5)")

    vocab_size = 248320

    torch.manual_seed(42)
    logits = torch.randn(vocab_size, dtype=torch.float16, device="cuda")

    # Full-range int32 bitmask (including negative values = bit 31 set)
    bitmask = torch.randint(-(2**31), 2**31 - 1, (vocab_size // 32,),
                            dtype=torch.int32, device="cuda")

    # Reference: vectorized PyTorch
    token_mask = expand_bitmask(bitmask, vocab_size)
    masked_logits = logits.float().clone()
    masked_logits[~token_mask] = float("-inf")
    ref_argmax = masked_logits.argmax().item()
    ref_density = token_mask.sum().item() / vocab_size

    # Triton
    tri_argmax, tri_density = fused_masked_argmax(logits, bitmask)

    match = tri_argmax == ref_argmax
    dens_match = abs(tri_density - ref_density) < 1e-5

    print(f"  PyTorch argmax: {ref_argmax}")
    print(f"  Triton argmax:  {tri_argmax}")
    print(f"  Match: {'✅' if match else '❌'}")
    print(f"  PyTorch density: {ref_density:.6f}")
    print(f"  Triton density:  {tri_density:.6f}")
    print(f"  Density match:   {'✅' if dens_match else '❌'}")

    assert match, f"Argmax mismatch: {tri_argmax} != {ref_argmax}"
    assert dens_match, f"Density mismatch: {tri_density} vs {ref_density}"
    print("  → PASS ✅")


# ===========================================================================
# Test 3: Edge cases
# ===========================================================================
def test_edge_cases():
    """All-valid, single-valid, none-valid."""
    print("\n▶ Test 3: Edge cases")

    # 3a: all valid
    vocab_size = 64
    logits = torch.randn(vocab_size, dtype=torch.float32, device="cuda")
    bitmask = torch.tensor([-1, -1], dtype=torch.int32, device="cuda")
    tid, dens = fused_masked_argmax(logits, bitmask)
    assert dens == 1.0, f"All-valid density should be 1.0, got {dens}"
    assert tid == logits.argmax().item()
    print(f"  3a All-valid:   argmax={tid}, density={dens:.2f} ✅")

    # 3b: only token 0 valid
    bitmask = torch.tensor([1, 0], dtype=torch.int32, device="cuda")
    tid, dens = fused_masked_argmax(logits, bitmask)
    assert tid == 0, f"Single-valid should give 0, got {tid}"
    assert abs(dens - 1/64) < 1e-5
    print(f"  3b Single-valid: argmax={tid}, density={dens:.4f} ✅")

    # 3c: negative int32 (bit 31 set) — ensure kernel handles signed ints
    bitmask = torch.tensor([-1, -1], dtype=torch.int32, device="cuda")
    tid, dens = fused_masked_argmax(logits, bitmask)
    assert dens == 1.0
    assert tid == logits.argmax().item()
    print(f"  3c Negative-int32: argmax={tid}, density={dens:.2f} ✅")


# ===========================================================================
# Test 3b: Batch mode correctness
# ===========================================================================
def test_batch_correctness():
    """Verify batch kernel matches single-row kernel for each row."""
    from velospec.triton.fused_logit_processor import fused_masked_argmax_batch

    print("\n▶ Test 3b: Batch correctness (B=8, vocab=248320)")

    vocab_size = 248320
    B = 8
    torch.manual_seed(99)

    logits = torch.randn(B, vocab_size, dtype=torch.float16, device="cuda")
    bitmask = torch.randint(-(2**31), 2**31 - 1,
                            (B, vocab_size // 32), dtype=torch.int32, device="cuda")

    # Reference: run single-row kernel for each row
    # ⚠️ Must sync+clone: _sync=False returns shared buffer refs that get overwritten
    ref_ids = []
    ref_valids = []
    for b in range(B):
        tid, dens = fused_masked_argmax(logits[b], bitmask[b], _sync=True)
        ref_ids.append(tid)
        ref_valids.append(int(dens * vocab_size))

    # Batch kernel
    batch_ids, batch_valids = fused_masked_argmax_batch(logits, bitmask)

    match = True
    for b in range(B):
        if batch_ids[b].item() != ref_ids[b]:
            match = False
    valids_match = True
    for b in range(B):
        if abs(batch_valids[b].item() - ref_valids[b]) > 1:
            valids_match = False

    print(f"  Argmax match:  {'✅' if match else '❌'}")
    print(f"  Valid match:   {'✅' if valids_match else '❌'}")
    if not match:
        for b in range(B):
            same = "✅" if batch_ids[b].item() == ref_ids[b] else "❌"
            print(f"    row {b}: batch={batch_ids[b].item()}, single={ref_ids[b]} {same}")

    assert match, "Batch argmax mismatch"
    assert valids_match, "Batch valid count mismatch"
    print("  → PASS ✅")


# ===========================================================================
# Test 4: Performance — Triton vs PyTorch (fair comparison)
# ===========================================================================
def test_performance():
    """Benchmark Triton kernel vs PyTorch mask+argmax baseline."""
    from velospec.triton.fused_logit_processor import fused_masked_argmax_batch

    print("\n▶ Test 4: Performance (vocab=248320)")

    vocab_size = 248320

    torch.manual_seed(0)
    logits = torch.randn(vocab_size, dtype=torch.float16, device="cuda")
    bitmask = torch.randint(-(2**31), 2**31 - 1, (vocab_size // 32,),
                            dtype=torch.int32, device="cuda")

    # Pre-compute PyTorch reference mask (not part of benchmark)
    token_mask = expand_bitmask(bitmask, vocab_size)

    N = 100

    # --- Warmup Triton (fills buffer cache + JIT compile) ---
    for _ in range(10):
        fused_masked_argmax(logits, bitmask)

    # --- Benchmark Triton (sync mode — current API) ---
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        fused_masked_argmax(logits, bitmask, _sync=True)
    torch.cuda.synchronize()
    triton_sync_us = (time.perf_counter() - t0) / N * 1e6

    # --- Benchmark Triton (async) ---
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        fused_masked_argmax(logits, bitmask, _sync=False)
    torch.cuda.synchronize()
    triton_async_us = (time.perf_counter() - t0) / N * 1e6

    # --- Benchmark PyTorch ---
    for _ in range(10):
        masked = logits.float().clone()
        masked[~token_mask] = float("-inf")
        _ = masked.argmax().item()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        masked = logits.float().clone()
        masked[~token_mask] = float("-inf")
        _ = masked.argmax().item()
    torch.cuda.synchronize()
    pytorch_us = (time.perf_counter() - t0) / N * 1e6

    print(f"  Single-row results:")
    print(f"    Triton (sync):   {triton_sync_us:.1f} μs/call")
    print(f"    Triton (async):  {triton_async_us:.1f} μs/call")
    print(f"    PyTorch:         {pytorch_us:.1f} μs/call")
    print(f"    Speedup (sync):  {pytorch_us / triton_sync_us:.1f}×")
    print(f"    Speedup (async): {pytorch_us / triton_async_us:.1f}×")

    # --- Benchmark BATCH mode (K+1 positions, like spec decoding verify phase) ---
    print(f"\n  Batch mode (simulates spec decoding verify phase):")

    for K_plus_1 in [2, 6, 8, 12]:
        batch_logits = torch.randn(K_plus_1, vocab_size, dtype=torch.float16, device="cuda")
        batch_bitmask = torch.randint(-(2**31), 2**31 - 1,
                                      (K_plus_1, vocab_size // 32),
                                      dtype=torch.int32, device="cuda")

        # PyTorch batch: loop K+1 times
        token_masks = [expand_bitmask(batch_bitmask[b], vocab_size) for b in range(K_plus_1)]

        # Warmup
        for _ in range(10):
            fused_masked_argmax_batch(batch_logits, batch_bitmask)

        # Triton batch
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N):
            fused_masked_argmax_batch(batch_logits, batch_bitmask)
        torch.cuda.synchronize()
        triton_batch_us = (time.perf_counter() - t0) / N * 1e6

        # PyTorch sequential (K+1 calls)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N):
            for b in range(K_plus_1):
                masked = batch_logits[b].float().clone()
                masked[~token_masks[b]] = float("-inf")
                _ = masked.argmax().item()
        torch.cuda.synchronize()
        pytorch_batch_us = (time.perf_counter() - t0) / N * 1e6

        speedup = pytorch_batch_us / triton_batch_us
        print(f"    K+1={K_plus_1:>2d}:  Triton {triton_batch_us:>7.1f} μs  |  "
              f"PyTorch {pytorch_batch_us:>7.1f} μs  |  {speedup:.1f}×")


# ===========================================================================
# Test 5: Multiple vocab sizes (portability)
# ===========================================================================
def test_multiple_sizes():
    """Ensure kernel works across different vocab sizes."""
    print("\n▶ Test 5: Multiple vocab sizes")

    for vocab_size in [128, 1024, 32000, 128000, 248320]:
        logits = torch.randn(vocab_size, dtype=torch.float32, device="cuda")
        num_words = (vocab_size + 31) // 32
        bitmask = torch.randint(0, 2**31, (num_words,),
                                dtype=torch.int32, device="cuda")

        token_mask = expand_bitmask(bitmask, vocab_size)
        masked = logits.clone()
        masked[~token_mask] = float("-inf")
        ref = masked.argmax().item()

        tri, _ = fused_masked_argmax(logits, bitmask)
        ok = "✅" if tri == ref else "❌"
        print(f"  vocab={vocab_size:>7d}: triton={tri}, ref={ref} {ok}")
        assert tri == ref, f"Mismatch at vocab={vocab_size}: {tri} != {ref}"

    print("  → PASS ✅")


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    test_known_answer()
    test_correctness_full()
    test_edge_cases()
    test_batch_correctness()
    test_performance()
    test_multiple_sizes()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✅")
    print("=" * 60)
    print("\nNext step: run ncu profiling (see README benchmark section)")
