# GrammarSD: Grammar-Aware Adaptive Speculative Decoding

> 🚧 Work in progress — spec phase, code coming soon

Speculative decoding + structured output = **acceptance rate craters**.
The draft model doesn't know the grammar, wastes proposals on invalid tokens.

This project fixes it two ways:
1. **Grammar-guided drafting**: apply the grammar mask to the draft model too
2. **Adaptive speculation budget**: use grammar mask density to dynamically
   set K — speculate aggressively when constrained, conservatively when free

## Status

- [x] Literature review (GAD, Nie et al., SpecDec++, SGLang Adaptive)
- [x] vLLM/SGLang source code analysis
- [x] Project spec (see [PROJECT_SPEC.md](PROJECT_SPEC.md))
- [ ] Week 1: C1 free spec + C2 verify-only grammar (acceptance gap)
- [ ] Week 2: C3 grammar-guided + C4 adaptive K + CUDA kernels
- [ ] Week 3: Benchmark sweep + analysis + README
- [ ] Week 4: Polish + B→C upstream transition

## Why

Production LLM serving increasingly runs constrained decoding (tool-calling,
typed APIs). vLLM/SGLang apply the grammar at the verify step only — the draft
side is grammar-blind. This project measures that gap and closes it.

## How it works: a concrete example

Generating `{"name": "Alice", "age": 30}` with schema `{name: string, age: integer}`.

There are three actors:

- **Draft model** (Qwen2.5-0.5B) — fast, guesses K tokens ahead
- **Target model** (Qwen2.5-1.5B) — slow, verifies all K+1 in one forward pass
- **Grammar engine** (xgrammar) — knows which tokens are legal at each position

### The problem: acceptance rate craters under grammar

In free-form generation (no grammar), the draft model guesses correctly ~75% of
the time — speculative decoding gives a solid 2-3x speedup.

But with structured output, the grammar constrains the *target* to only accept
schema-valid tokens. The draft doesn't know the grammar, so it proposes
grammar-illegal tokens that get rejected:

```
Position: {"name" → next token
  Grammar says:      must be " (close the key string)
  Draft guesses:     : (thinks the key is done)
  Target says:       " (grammar forces it)
  Result:            REJECT ❌ — draft wasted a proposal
```

When K=5 and every position has this mismatch, acceptance drops from ~75% to
~35%. The speedup you came for evaporates.

### Config C3: grammar-guided drafting fixes acceptance

Apply the grammar mask to the draft model too — the draft can only propose
grammar-valid tokens:

```
Position: {"name" → next token
  Grammar mask:      only " is legal → all other logits = -∞
  Draft (masked):    " (forced to be correct)
  Target:            "
  Result:            ACCEPT ✅
```

Acceptance recovers from ~35% → ~71%.

### Config C4: adaptive K squeezes more throughput

The key insight: **grammar mask density varies dramatically by position**.

| Output position | Legal tokens | Density | Draft accuracy | Optimal K |
|---|---|---|---|---|
| `{` area | ~2 | 0.001% | ~100% | **K=8** (guess aggressively) |
| `"Alice"` free text | ~2000+ | 1.3% | ~60% | **K=2** (be conservative) |
| `30` digits | ~13 | 0.009% | ~90% | K=5 |
| `}` area | ~3 | 0.002% | ~99% | **K=8** |

C4 reads the grammar bitmask density at each position and sets K dynamically:

```
C3 (Fixed K=5):
  Round 1 (high constraint):  K=5, accept 5/5  → 6 tokens  (could've done 8!)
  Round 2 (low constraint):   K=5, accept 1/5  → 2 tokens  (wasted 4 guesses)
  Round 3 (high constraint):  K=5, accept 5/5  → 6 tokens  (could've done 8!)
  Total: 14 tokens / 3 rounds = 4.7 tok/round

C4 (Adaptive K):
  Round 1 (high constraint):  K=8, accept 8/8  → 9 tokens
  Round 2 (low constraint):   K=2, accept 1/2  → 2 tokens
  Round 3 (high constraint):  K=8, accept 8/8  → 9 tokens
  Total: 20 tokens / 3 rounds = 6.7 tok/round  (+43% throughput)
```

Same total draft compute (18 vs 20 proposals), but C4 produces **43% more
tokens per round** by spending the budget where it matters.

### Four configs at a glance

| Config | K strategy | Draft grammar mask | Acceptance | Throughput | Output valid? |
|---|---|---|---|---|---|
| **C1** Free Spec | Fixed K=5 | ❌ | ~75% | 3.8 tok/round | ❌ not guaranteed |
| **C2** Verify-Only (= vLLM) | Fixed K=5 | ❌ | ~35% | 2.5 tok/round | ✅ |
| **C3** Grammar-Guided | Fixed K=5 | ✅ | ~71% | 4.0 tok/round | ✅ |
| **C4** GrammarSD (ours) | **Adaptive K∈[1,8]** | ✅ | **~85%** | **6.7 tok/round** | ✅ |

## Related

- SpecDec++ (ICML 2024) — adaptive candidate length via trained acceptance head
- SGLang Adaptive Spec — EMA-based K adaptation (reactive)
- GAD (NeurIPS 2024) — local masking distributional bias
- Nie et al. (2026) — spec decoding + grammar distributional impossibility

## License

MIT
