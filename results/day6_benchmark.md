# Week 1 Day 6: Benchmark C1 + C2 (Post Fix)

**Date:** 2025-06-29 (re-run after accept rate fix)
**GPU:** Colab Pro A100-SXM4-40GB
**Models:** Qwen3.5-0.8B (draft) + Qwen3.5-4B (target)
**K=5**, max_tokens=256

## Fix Applied

`total_drafted` moved from before verification to after, only counting tokens
actually verified. Previously, tokens never checked (due to reject-break or
grammar early-termination) counted as "rejected", artificially depressing
accept rate for BOTH C1 and C2.

## Raw Summary

```
Config Schema         Avg Accept    Avg TPS    Avg Len
------------------------------------------------------
C1     nested             77.02%        6.1        258
C1     simple             77.02%        4.1        258
C1     tool_call          77.02%        6.1        258
C2     nested             77.64%        5.2        179
C2     simple             72.54%        4.2         25
C2     tool_call          51.58%        3.5         30
```

## Key Findings

### C1 (free speculative decoding)
- Accept rate 77.02% — consistent across all schemas (correct, C1 ignores grammar)
- Up from 48.66% (old measurement counted unverified reject-break tokens)

### C2 (verify-only grammar)
- **nested: 77.64% > C1 77.02%** — grammar mask narrows target candidate space,
  draft hit rate slightly improves. Long output (179 tok) means plenty of
  free-form string values where grammar is loose.
- **simple: 72.54% < C1** — small gap, short output (25 tok), mostly structural tokens
- **tool_call: 51.58% << C1 77.02%** — **THIS IS THE GAP**. Draft generates
  unconstrained tokens that violate tool_call enum/function-name constraints.
  Grammar mask rejects them. 25.4 percentage point drop = the cost of
  verify-only grammar on constrained schemas.

### C2 early termination (unchanged)
- simple: ~25 tokens, tool_call: ~30 tokens, nested: ~179 tokens
- Grammar `is_terminated()` fires when JSON is complete — feature, not bug

### The Gap (C1 → C2) — motivates C3
| Schema | C1 Accept | C2 Accept | Gap |
|--------|-----------|-----------|-----|
| nested | 77.02% | 77.64% | -0.62% (negligible) |
| simple | 77.02% | 72.54% | 4.48% |
| tool_call | 77.02% | 51.58% | **25.44%** |

C3 (grammar mask on draft + target) should recover tool_call accept rate
by preventing draft from generating grammar-illegal tokens in the first place.

## Next Steps
- [x] Day 6 benchmark re-run with fix ✅
- [x] Day 7: tag v0.1-gap-measured ✅
- [ ] Week 2: C3 (draft grammar) + CUDA kernels
