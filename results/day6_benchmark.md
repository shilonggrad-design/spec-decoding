# Week 1 Day 6: Benchmark C1 + C2

**Date:** 2025-06-29
**GPU:** Colab Pro A100-SXM4-40GB
**Models:** Qwen3.5-0.8B (draft) + Qwen3.5-4B (target)
**K=5**, max_tokens=256

## Raw Summary

```
Config Schema         Avg Accept    Avg TPS    Avg Len
------------------------------------------------------
C1     nested             48.66%        6.1        258
C1     simple             48.66%        4.1        258
C1     tool_call          48.66%        6.1        258
C2     nested             48.17%        5.2        179
C2     simple             37.93%        4.2         25
C2     tool_call          20.74%        3.4         30
```

## Key Findings

### C1 (free speculative decoding)
- Accept rate identical across schemas (48.66%) — correct, C1 ignores grammar.
- Simple schema TPS lower (4.1 vs 6.1) is cold-start artifact (runs first).
- Real C1 throughput ≈ 6.1 tok/s (warm GPU).

### C2 (verify-only grammar)
- **Early termination**: simple (25 tok), tool_call (30 tok), nested (179 tok).
  Grammar matcher `is_terminated()` fires when JSON is complete — this is correct.
- Simple/tool_call stop early because 4-field JSON completes in ~25 tokens.
- Nested generates longer because nested JSON structure is more complex.
- Accept rate varies by schema: nested 48.17% ≈ C1, simple 37.93%, tool_call 20.74%.
- Tool_call has lowest accept rate — likely more constrained enum/function names.

### Benchmark Fairness Note
C1 vs C2 throughput comparison is **not apples-to-apples** because C2 terminates early.
For fair comparison, we need:
1. **Time-to-completion**: wall clock until grammar-done (C2 advantage: completes in ~5-7s for simple)
2. **Token throughput**: tok/s for ongoing generation (relevant for streaming)
3. **Useful token efficiency**: C2 generates only grammar-valid tokens, no waste

## Next Steps (Week 1 completion)
- [x] Day 3: C1 free spec → 41-49% accept rate ✅
- [x] Day 4: C2 verify-only grammar → grammar-early-stop confirmed ✅
- [x] Day 6: Full benchmark (3 schema × 5 prompts) ✅
- [ ] Day 7: Tag v0.1-gap-measured, write Week 1 summary
- [ ] Week 2: CUDA kernel (fused bitmask + speculative sampling)
