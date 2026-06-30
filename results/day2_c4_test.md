# Day 2: C4 Adaptive K Test — tool_call schema

**Date**: 2026-06-30
**Config**: C4 (GrammarSD full: C3 grammar-guided draft + adaptive K)
**Schema**: tool_call
**Prompt**: "Call a function to search for AI news."
**Models**: Qwen3.5-4B (target) + Qwen3.5-0.8B (draft), vocab=248,320

## Results

| Metric | Value |
|--------|-------|
| Accept rate | **100.00%** |
| Tokens generated | 16 |
| Time | 1.838s |
| Rounds | 2 |
| Throughput | 8.7 tok/s |

## K Trace

| Round | Density | K chosen | Interpretation |
|-------|---------|----------|----------------|
| 1 | 8.05e-06 | 8 | Extremely low — `{` or `"function"` key area, ~2 valid tokens |
| 2 | 1.51e-03 | 8 | Still very low — enum value `"search"` area, ~375 valid tokens |

Both rounds speculated with maximum K=8. This is correct behavior:
- tool_call schema has tight enum constraints → density is extremely low at every position
- Low density → draft can't miss → speculate aggressively (K=8)

## Output (valid JSON ✅)

```json
{"function": "search", "arguments": {"query": "AI news"}}
```

## Analysis

1. **C4 acceptance = 100%**: Same as C3. Grammar-guided draft ensures all draft tokens are grammar-valid → target accepts everything.

2. **Adaptive K = 8 throughout**: For tool_call schema, density is always < 0.005 threshold, so K always maxes out. This is the optimal behavior — every position is so constrained that the draft model essentially can't miss.

3. **Throughput**: 16 tokens in 2 rounds = 8 tokens/round. With K=5 (fixed C3), this would have been at most 6 tokens/round → C4 gives **33% more throughput per round** on this schema.

4. **Contrast with free-form schemas**: On simple/nested schemas, we expect to see K vary more — K=8 at key boundaries, K=1 at free-form string values. The benchmark (Day 3) will reveal this.
