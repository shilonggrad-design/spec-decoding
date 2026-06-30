# Week 2 Results: C3/C4 Benchmark + CUDA Kernels

> Fill this in after running Day 1-7. Replace TBD with actual numbers.

## Environment

- **GPU**: TBD (run `nvidia-smi`)
- **CUDA**: TBD (run `nvcc --version`)
- **Models**: Qwen3.5-4B (target) + Qwen3.5-0.8B (draft)
- **vocab_size**: 248,320
- **Date**: TBD

---

## Day 1: C3 Quick Test (tool_call schema)

| Config | Accept Rate | Tokens | Time | Output Valid? |
|--------|-------------|--------|------|---------------|
| C1 | TBD | TBD | TBD | N/A |
| C2 | TBD | TBD | TBD | ✅ |
| C3 | TBD | TBD | TBD | ✅ |

**Expected**: C3 should recover to ~100% (Week 1 confirmed).

---

## Day 2: C4 Quick Test (tool_call schema)

| Metric | Value |
|--------|-------|
| Accept rate | TBD |
| Tokens generated | TBD |
| Time | TBD |
| K trace | TBD |

**What to look for in k_trace**:
- Low density positions (JSON keys, brackets) → K=8
- High density positions (free text values) → K=1
- Moderate density → K=4

---

## Day 3: Full Benchmark (C1-C4 × 3 schemas × 5 prompts = 60 trials)

### Acceptance Rate Summary

| Config | simple | tool_call | nested |
|--------|--------|-----------|--------|
| C1 | TBD | TBD | TBD |
| C2 | TBD | TBD | TBD |
| C3 | TBD | TBD | TBD |
| C4 | TBD | TBD | TBD |

### Throughput (tokens/sec) Summary

| Config | simple | tool_call | nested |
|--------|--------|-----------|--------|
| C1 | TBD | TBD | TBD |
| C2 | TBD | TBD | TBD |
| C3 | TBD | TBD | TBD |
| C4 | TBD | TBD | TBD |

### Gap Analysis

| Schema | C1→C2 gap | C2→C3 recovery | C3→C4 delta |
|--------|-----------|-----------------|-------------|
| simple | TBD | TBD | TBD |
| tool_call | TBD | TBD | TBD |
| nested | TBD | TBD | TBD |

### C4 Adaptive K Analysis

| Schema | Avg Density | Avg K | K=1 freq | K=4 freq | K=8 freq |
|--------|-------------|-------|----------|----------|----------|
| simple | TBD | TBD | TBD | TBD | TBD |
| tool_call | TBD | TBD | TBD | TBD | TBD |
| nested | TBD | TBD | TBD | TBD | TBD |

**CSV**: `results/week2_c1_c4.csv`

---

## Day 4: CUDA Kernel Validation

```
# Paste test_cuda_kernels.py output here
```

| Test | Result |
|------|--------|
| Kernel 1: popcount (basic) | TBD |
| Kernel 1: popcount (sparse) | TBD |
| Kernel 2: argmax (single) | TBD |
| Kernel 2: argmax (batch K+1) | TBD |
| Kernel 3: fused_sample (greedy) | TBD |
| Kernel 3: fused_sample (sampling) | TBD |
| Speedup: popcount Python→CUDA | TBD × |

---

## Key Findings

1. **C3 recovers the acceptance gap**: TBD
2. **C4 adaptive K effectiveness**: TBD
3. **CUDA kernel speedup**: TBD
4. **Schema-dependent behavior**: TBD

---

## Comparison: Week 1 vs Week 2

| Metric | Week 1 (C1/C2) | Week 2 (C3/C4) |
|--------|----------------|----------------|
| Best accept rate (tool_call) | 51.58% (C2) | TBD |
| Throughput improvement | baseline | TBD |
| Output validity guaranteed | C2 only | C3+C4 |
