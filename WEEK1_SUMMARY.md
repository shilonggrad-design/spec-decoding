# Week 1 Summary — Grammar-Aware Speculative Decoding

**Tag:** v0.1-gap-measured
**Date:** 2025-06-29
**GPU:** Colab Pro A100-SXM4-40GB, CUDA 13.0

---

## Goal

Measure the gap between free speculative decoding (C1) and grammar-constrained speculative decoding (C2), establishing baseline for Week 2–3 CUDA kernel optimization.

## Models

| Role | Model | Params | VRAM |
|------|-------|--------|------|
| Draft | Qwen3.5-0.8B | 0.8B | ~1.6 GB |
| Target | Qwen3.5-4B | 4B | ~8 GB |
| **Vocab size** | 248,320 tokens | | |
| **Bitmask** | 7,761 × int32 ≈ 31 KB | (fits L2) | |

## Configurations Tested

| Config | Draft Grammar | Verify Grammar | K |
|--------|:-------------:|:--------------:|:-:|
| C1 | ✗ | ✗ | 5 |
| C2 | ✗ | ✓ | 5 |
| C3 | ✓ | ✓ | 5 | *(Week 2)* |
| C4 | ✓ | ✓ | adaptive | *(Week 3)* |

## Results

### C1 — Free Speculative Decoding (baseline)
- **Accept rate:** 48.66% (consistent across all schemas)
- **Throughput:** ~6.1 tok/s (A100, warm)
- **Output length:** 256–261 tokens (max_tokens=256)
- Grammar has zero effect — confirm C1 ignores schema parameter correctly.

### C2 — Verify-Only Grammar
| Schema | Avg Accept | Avg Output Len | Avg Time |
|--------|-----------|----------------|----------|
| simple | 37.93% | **25 tok** | ~6.2s |
| tool_call | 20.74% | **30 tok** | ~8.3s |
| nested | 48.17% | **179 tok** | ~32.8s |

**Key insight:** C2 terminates early when grammar determines JSON is complete. Simple 4-field JSON finishes in ~25 tokens — this is a feature, not a bug. Time-to-completion is the right metric, not throughput.

### C2 Density Trace (simple schema, excerpt)
```
pos 0-3:   0.001%  ← JSON structural tokens (keys, braces)
pos 4-6:   98.9%   ← Free-form string values
pos 7-11:  0.15%   ← Key names (enum-constrained)
pos 12-13: 0.15%   ← Numeric values
pos 18-19: 98.9%   ← Another string value
...
Average:   37.1%    ← High variance: 0.001% ↔ 98.9%
```

This extreme variance is the motivation for C4 adaptive K.

## Bugs Found & Fixed

1. **Off-by-one in logits indexing** — `logits[pos]` predicts `token[pos+1]` (causal LM property). Fixed to `prefix_len - 1 + i`.
2. **In-place mutation** — `apply_token_bitmask_inplace` polluted subsequent positions. Fixed with `.clone()`.
3. **File corruption from repeated patches** — Rewrote speculator.py from scratch.

## Deliverables

- `src/baseline.py` — Vanilla AR + Grammar AR baseline
- `src/speculator.py` — C1/C2 speculative decoding with xgrammar
- `src/bench_week1.py` — Benchmark runner (3 schema × 5 prompts)
- `src/test_validity.py` — Correctness tests (Week 2)
- `schemas/` — 3 JSON schemas (simple, tool_call, nested)
- `results/` — Day 2 baseline + Day 6 benchmark

## Next: Week 2 — C3 (Draft+Verify Grammar) + CUDA Kernel

1. Implement C3: grammar bitmask in draft phase too
2. Write fused CUDA kernel: `apply_bitmask_and_sample` (single pass)
3. Profile with `nsight` on Vast.ai RTX 3090
4. Compare C3 vs C2 accept rate improvement
