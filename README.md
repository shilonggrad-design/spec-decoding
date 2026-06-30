# GrammarSD: Grammar-Aware Adaptive Speculative Decoding

> Active development — Week 1 complete, Week 2 code ready (C3 verified, C4+CUDA pending validation)

Speculative decoding + structured output = **acceptance rate craters**.
The draft model doesn't know the grammar, wastes proposals on invalid tokens.

This project fixes it two ways:
1. **Grammar-guided drafting**: apply the grammar mask to the draft model too
2. **Adaptive speculation budget**: use grammar mask density to dynamically
   set K — speculate aggressively when constrained, conservatively when free

## Status

- [x] Literature review (GAD, Nie et al., SpecDec++, SGLang Adaptive)
- [x] vLLM/SGLang source code analysis (source-verified from `main` branch)
- [x] Project spec (see [PROJECT_SPEC.md](PROJECT_SPEC.md))
- [x] Week 1: C1 free spec + C2 verify-only grammar — acceptance gap measured and root-caused
- [x] Week 2: C3 grammar-guided + C4 adaptive K + CUDA kernels — code complete
- [ ] Week 3: Nsight profiling + optimization sweep
- [ ] Week 4: Polish + upstream transition plan

## Models

- **Draft model**: Qwen3.5-0.8B (~1.6 GB)
- **Target model**: Qwen3.5-4B (~8 GB)
- **Grammar engine**: xgrammar (per-position token bitmask)
- **vocab_size**: 248,320 — bitmask = ceil(248320/32) = 7,761 × int32 ≈ 31 KB

## Week 1 Results (post-fix benchmark, [data截至 2026-06-29])

| Config | Schema | Accept Rate | Notes |
|--------|--------|-------------|-------|
| C1 (no grammar) | all identical | **77.02%** | Baseline — no grammar anywhere |
| C2 (verify-only) | nested | **77.64%** | Surprisingly high — most tokens are string values |
| C2 (verify-only) | simple | **72.54%** | Moderate constraint |
| C2 (verify-only) | tool_call | **51.58%** | **The gap** — enum + format constraints kill draft accuracy |
| C3 (grammar-guided) | tool_call | **100.00%** | Draft constrained → target accepts everything |

**Key finding**: The acceptance gap is schema-dependent, not uniform. Tool-call
schemas (tight enum constraints) show a **-25.4% gap** between C1 and C2. C3
fully recovers it.

## How it works: a concrete example

Generating `{"function": "search", "arguments": {"query": "AI news"}}` with a tool-call schema.

There are three actors:

- **Draft model** (Qwen3.5-0.8B) — fast, guesses K tokens ahead
- **Target model** (Qwen3.5-4B) — slow, verifies all K+1 in one forward pass
- **Grammar engine** (xgrammar) — knows which tokens are legal at each position

### The problem: acceptance rate craters under grammar

In free-form generation (no grammar), the draft model guesses correctly ~77% of
the time — speculative decoding gives a solid 2-3x speedup.

But with structured output, the grammar constrains the *target* to only accept
schema-valid tokens. The draft doesn't know the grammar, so it proposes
grammar-illegal tokens that get rejected:

```
Position: {"function" → next token
  Grammar says:      must be " (close the key string)
  Draft guesses:     : (thinks the key is done)
  Target says:       " (grammar forces it)
  Result:            REJECT ❌ — draft wasted a proposal
```

For tool-call schemas, acceptance drops from 77% to 52%. The speedup you came
for is cut nearly in half.

### Config C3: grammar-guided drafting fixes acceptance

Apply the grammar mask to the draft model too — the draft can only propose
grammar-valid tokens:

```
Position: {"function" → next token
  Grammar mask:      only " is legal → all other logits = -∞
  Draft (masked):    " (forced to be correct)
  Target:            "
  Result:            ACCEPT ✅
```

Acceptance recovers to **100%** on tool_call schema (verified).

### Config C4: adaptive K squeezes more throughput

The key insight: **grammar mask density varies dramatically by position**.

| Output position | Legal tokens | Density | Draft accuracy | Optimal K |
|---|---|---|---|---|
| `{` area | ~2 | 0.001% | ~100% | **K=8** (guess aggressively) |
| `"AI news"` free text | ~2000+ | 1.3% | ~60% | **K=1** (be conservative) |
| `30` digits | ~13 | 0.009% | ~90% | **K=8** |
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
  Round 2 (low constraint):   K=1, accept 1/1  → 1 tokens
  Round 3 (high constraint):  K=8, accept 8/8  → 9 tokens
  Total: 19 tokens / 3 rounds = 6.3 tok/round  (+34% throughput)
```

C4 spends compute where it matters — speculating aggressively at high-constraint
positions (JSON keys, braces, enums) and conservatively at low-constraint
positions (free-form string values).

### Four configs at a glance

| Config | K strategy | Draft grammar mask | Target grammar mask | Output valid? |
|---|---|---|---|---|
| **C1** Free Spec | Fixed K=5 | ❌ | ❌ | ❌ not guaranteed |
| **C2** Verify-Only (= vLLM/SGLang) | Fixed K=5 | ❌ | ✅ | ✅ |
| **C3** Grammar-Guided | Fixed K=5 | ✅ | ✅ | ✅ |
| **C4** GrammarSD (ours) | **Adaptive K∈[1,8]** | ✅ | ✅ | ✅ |

### CUDA Kernels (Week 2)

Three custom kernels accelerate the hot paths:

| Kernel | File | Purpose | HPC Technique |
|--------|------|---------|---------------|
| 1 | `popcount_density.cu` | Count valid tokens in bitmask | `__popc` intrinsic + warp shuffle reduction |
| 2 | `grammar_masked_argmax.cu` | Fused mask+argmax for verify | One block per position, parallel argmax |
| 3 | `fused_sample.cu` | Fused mask+softmax+sample for draft | Online softmax, 5× memory traffic reduction |

## Related

- SpecDec++ (ICML 2024) — adaptive candidate length via trained acceptance head
- SGLang Adaptive Spec — EMA-based K adaptation (reactive)
- GAD (NeurIPS 2024) — local masking distributional bias
- Nie et al. (2026) — spec decoding + grammar distributional impossibility

## License

MIT
