# GrammarSD: Grammar-Aware Adaptive Speculative Decoding — Project Spec (v3)

**Public CUDA + ML Systems portfolio project for AI/ML Infra + HPC/Kernel job search.**
**Direction**: Adaptive speculative decoding driven by grammar mask density for structured output generation.
**Budget**: ~3-4 weeks @ 2 hr/day weekday + weekends = 30-50 hours total.
**Owner**: Shilong Zhang. **Created**: 2026-06-26. **Supersedes**: v2 (grammar-guided drafting only).
**Repo target**: `github.com/<your-handle>/grammar-sd`

---

## 0. The hook (what makes this NOT "another spec decoding repo")

Speculative decoding accelerates LLM inference by having a small draft model guess K tokens, then verifying them in one target model forward pass. The speedup depends on **acceptance rate** — how many draft tokens the target accepts.

Existing adaptive spec decoding (SpecDec++, AdaSpec, SGLang Adaptive) adjusts K using **backward-looking signals**: past acceptance rate EMA, trained prediction heads, or system load. They react to rejections *after* they happen.

**This project uses a forward-looking, zero-cost signal: grammar mask density.**

When generating structured output (JSON, tool calls), a grammar engine produces a per-position bitmask of valid tokens. The **density** of that bitmask (valid_tokens / vocab_size) deterministically predicts draft accuracy:

```
Position type              Valid tokens    Density      Draft accuracy
─────────────────────────────────────────────────────────────────────
JSON key boundary             ~2            0.00002      ~100%
String value interior         ~100          0.0008       ~95%
Numeric value                 ~15           0.0001       ~99%
Free-form text value          ~5000+        0.04         ~60%
```

Low density → draft can't miss → **increase K, speculate aggressively**.
High density → draft will miss → **decrease K, don't waste compute**.

No existing work uses grammar density as the speculation-length signal. This project implements it, benchmarks it against fixed-K baselines, and ships custom CUDA kernels for the density computation and grammar-masked sampling.

---

## 0.1 Competitive landscape (source-verified 2026-06-26)

### Adaptive K in existing systems

| System | Year | K Signal | Granularity | Training? | Forward-looking? |
|---|---|---|---|---|---|
| SGLang Adaptive | 2025 | Past acceptance rate EMA | Per-batch | ❌ | ❌ Backward-looking |
| SpecDec++ | ICML 2024 | Trained acceptance prediction head | Per-step | ✅ | ✅ But requires training |
| AdaSpec | 2025 | Request load + SLO | System-level | ❌ | ❌ External signal |
| SpecRouter | 2025 | Token distribution divergence | Per-step | ❌ | ✅ |
| **GrammarSD (this project)** | — | **Grammar mask density** | **Per-position** | **❌** | **✅ Deterministic** |

Sources:
- SGLang Adaptive: [sgl-project.github.io](https://sgl-project.github.io/advanced_features/adaptive_speculative_decoding.html) 🟢 — candidate_steps [1,3,7], EMA alpha=0.2, post-verify adjustment
- SpecDec++ (ICML 2024): [arxiv.org/abs/2405.19715](https://arxiv.org/abs/2405.19715), [icml.cc/virtual/2024/39629](https://icml.cc/virtual/2024/39629) 🟢 — MDP formulation, threshold policy optimal, trained acceptance head
- AdaSpec: [arxiv.org/abs/2503.05096](https://arxiv.org/abs/2503.05096) 🟡 — 66% speedup vs SOTA, SLO-driven
- SpecRouter: [arxiv.org/abs/2505.07680](https://arxiv.org/abs/2505.07680) 🟡 — multi-level speculative, token divergence routing

### Why grammar density is a better signal

1. **Zero-cost**: grammar mask is already computed for structured output — density = popcount / vocab_size
2. **Deterministic**: no ML model, no EMA lag — the signal is exact for the current position
3. **Forward-looking**: knows the constraint *before* drafting, not *after* rejection
4. **Per-position granularity**: adjusts K at every token, not per-batch or per-request

SpecDec++ proved (Theorem 1) that the optimal policy is a **threshold policy**: stop speculating when rejection probability exceeds a threshold. Grammar density is a direct, deterministic estimator of that threshold — no training needed.

### Grammar mask + spec decoding in production (status quo)

**vLLM** (source-verified from `main` branch, 2026-06-26):
- `gpu_model_runner.py` L4465: `apply_grammar_bitmask()` on target logits only — never on draft
- `spec_decode/llm_base_proposer.py` (1765 lines): `grep -i "grammar\|structured\|constraint\|mask"` → **0 matches**
- `structured_output/__init__.py` L204-300: generates per-position masks for target verification via `grammar.accept_tokens()` + `grammar.rollback()`
- `num_speculative_tokens` is a **global constant** — no adaptive adjustment exists
- 2 production bugs: [#27210](https://github.com/vllm-project/vllm/issues/27210) (EAGLE + structured output FSM crash), [#34650](https://github.com/vllm-project/vllm/issues/34650) (grammar mask silently not applied under spec decode + reasoning)

**SGLang** (source-verified from `main` branch, 2026-06-26):
- `managers/scheduler.py` L1627: `"We do not support overlap + spec + grammar yet"` — explicitly disables async scheduling
- Adaptive spec decoding exists but uses EMA of past acceptance, not grammar density
- Draft model path: no grammar-related code

**Key takeaway**: Production systems treat `num_speculative_tokens` as static (vLLM) or adapt it via backward-looking EMA (SGLang). Nobody uses the grammar bitmask density as a forward-looking signal.

---

## 0.2 Honesty boundary

### What's novel vs what's not

| Claim | Status |
|---|---|
| "Grammar density as K-scheduling signal" | ✅ Novel — no prior work uses this signal |
| "Adaptive K improves spec decoding" | ❌ NOT novel — SpecDec++ proved this (ICML 2024) |
| "Threshold policy is optimal for K selection" | ❌ NOT novel — SpecDec++ Theorem 1 |
| "Grammar mask on draft model" | ✅ Gap in vLLM/SGLang (verified from source) |
| "Fixing distributional bias of local masking" | ❌ Out of scope — GAD (NeurIPS 2024) + Nie et al. (2026) cover this |

### What this project IS

- An **empirical systems study**: does grammar-density-driven adaptive K beat fixed K and EMA-based adaptive K on structured output workloads?
- A **systems contribution**: custom CUDA kernels for density computation + fused grammar-masked sampling
- A **practical mitigation**: production teams losing spec decode speedup on JSON/tool-calling workloads can adopt this

### What this project IS NOT

- A new theorem (SpecDec++ already proved threshold policy optimality)
- A fix for distributional bias (GAD/Nie et al. own that problem)
- A production serving system (no continuous batching, no multi-GPU)

### Interview defense (memorize)

**Q: "SpecDec++ already proved adaptive K is optimal. What's new?"**

> "SpecDec++ proved the threshold policy is optimal and trained a prediction head to estimate rejection probability. My project uses the same threshold-policy framework but replaces the trained head with grammar mask density — a deterministic, zero-cost signal available in any structured-output pipeline. The contribution is showing that this grammar-specific signal is as effective as a trained predictor for K adaptation on structured workloads, while requiring no training data and adding < 1μs per position."

**Q: "vLLM/SGLang already do adaptive speculative decoding."**

> "SGLang's adaptive mode uses exponential moving average of past acceptance rate — it's backward-looking, adjusting per-batch with candidate tiers [1,3,7]. Grammar density is forward-looking and per-position. When the grammar transitions from a constrained JSON key boundary to a free-form string value, my system knows to reduce K before the draft wastes compute on rejections. SGLang's EMA won't catch up for several batches. Also, vLLM doesn't do adaptive K at all — num_speculative_tokens is a global constant."

---

## 1. Scope

### In scope ✅

- Speculative decoding (one draft + one target model), greedy decoding
- Grammar-constrained generation via `xgrammar` (produces per-position token bitmasks)
- **Four configurations** (see §3)
- Three custom CUDA kernels
- Adaptive K driven by grammar mask density (popcount-based, per-position)
- Empirical study: acceptance rate, speedup, wasted speculation across configs × schema complexity × K
- Correctness: 100% of outputs are grammar-valid
- Nsight Compute profiling
- README + benchmark writeup

### Out of scope ❌

- Writing a grammar/CFG engine (use xgrammar)
- Fixing distributional bias of local masking (GAD/Nie et al. problem)
- Batched serving, continuous batching, async scheduling
- Multi-GPU / tensor parallelism
- Quantization
- Custom attention kernel
- Tree-based speculation
- Trained acceptance prediction head (use deterministic grammar density instead)

### Stretch goals 🚀

- **B→C transition**: contribute a fix for vLLM issue #34650 (grammar mask not applied under spec decode + reasoning)
- **B→C transition**: upstream adaptive-K PR to SGLang (their adaptive infra already exists, add grammar-density signal)
- Φ-estimation experiment: measure TV distance vs Nie et al.'s theoretical prediction

---

## 2. Why this is feasible in 3-4 weeks

The novel code is small and layered on standard components:

| Component | Build vs Reuse | Effort |
|---|---|---|
| Spec decoding loop | Build (standard algorithm, ~200 lines Python) | 2 days |
| Grammar engine | Reuse (`xgrammar` library) | 0 |
| Grammar mask → density | Build (1 CUDA kernel, popcount) | 1 day |
| Adaptive K controller | Build (~50 lines Python, threshold logic) | 1 day |
| Grammar-masked sampling | Build (1 fused CUDA kernel) | 2 days |
| Grammar-masked acceptance | Build (1 CUDA kernel) | 2 days |
| Benchmark harness | Build (4 configs × schemas × K sweep) | 2 days |
| README + analysis | Write | 2 days |

Total estimated: ~12-14 working days at 2-3 hr/day = 3-4 weeks.

---

## 3. Technical design

### Models + grammar

- **Target**: `Qwen/Qwen2.5-1.5B-Instruct` (FP16, ~3GB)
- **Draft**: `Qwen/Qwen2.5-0.5B-Instruct` (FP16, ~1GB)
- **Grammar library**: `xgrammar` (fast, vLLM-adjacent, produces token bitmasks)
- **Schemas**: JSONSchemaBench subsets (simple Person schema → nested tool-call schema → complex nested objects)

### The four configs (the heart of the study)

| Config | K strategy | Draft grammar mask | Description | Expected |
|---|---|---|---|---|
| **C1: Free Spec** | Fixed K=5 | ❌ | = vLLM current behavior (no grammar) | ~75% accept, ~2.5x |
| **C2: Verify-Only Grammar** | Fixed K=5 | ❌ | Grammar mask on target only (= vLLM with structured output) | ~38% accept, ~1.3x — **exposes the gap** |
| **C3: Grammar-Guided Draft** | Fixed K=5 | ✅ | Grammar mask on draft + target | ~71% accept, ~2.4x — **recovers acceptance** |
| **C4: GrammarSD (Full)** | **Adaptive K∈[1,8]** | ✅ | Grammar mask on both + density-driven K | **~2.8-3.2x** — **best throughput** |

### Ablation analysis

- **C1 → C2**: cost of adding grammar to structured output (the gap)
- **C2 → C3**: value of grammar-guided drafting (acceptance recovery)
- **C3 → C4**: value of adaptive K on top of grammar mask (throughput optimization)
- **C1 → C4**: total improvement

### Adaptive K algorithm

```python
def compute_adaptive_K(grammar_mask, vocab_size, K_min=1, K_max=8, density_threshold=0.005):
    """
    Compute how many tokens to speculate based on grammar mask density.

    Low density (few valid tokens) → draft accuracy high → speculate more
    High density (many valid tokens) → draft accuracy low → speculate less

    This is the threshold policy from SpecDec++ (ICML 2024), with grammar
    density as the rejection-probability proxy instead of a trained head.
    """
    density = popcount(grammar_mask) / vocab_size   # CUDA kernel, < 1μs

    if density < density_threshold:
        # Highly constrained position: draft almost certainly correct
        K = K_max
    elif density < density_threshold * 4:
        # Moderately constrained
        K = (K_min + K_max) // 2
    else:
        # Loose constraint: draft likely wrong on some tokens
        K = K_min

    return K
```

**Why this works** (grounded in SpecDec++ theory):

SpecDec++ Theorem 1: the optimal stopping policy is a threshold on rejection probability. Grammar density is a **deterministic upper bound** on the space where draft can diverge from target:

- If only 2 tokens are valid (density = 0.00002), draft's probability of picking the same one as target is ~99%+ → rejection probability < threshold → continue speculating
- If 5000 tokens are valid (density = 0.04), draft has many ways to diverge → rejection probability > threshold → stop speculating

No trained predictor needed. The grammar itself is the predictor.

### Architecture diagram

```
                    ┌──────────────────────────────────────┐
                    │     Grammar State Tracker (CPU)        │
                    │                                       │
                    │  Per draft step:                      │
                    │  1. mask_i = grammar.get_mask()       │
                    │  2. density = popcount(mask_i) / V    │ ← Kernel 1
                    │  3. K_i = adaptive_K(density)         │
                    └──────────┬───────────────────────────┘
                               │ mask_i + K decision
                               ▼
                    ┌──────────────────────────────────────┐
                    │     Draft Model (GPU)                 │
                    │     autoregressive loop:              │
                    │                                       │
                    │  for i in range(K_adaptive):          │
                    │    logits = draft.forward()           │
                    │    fused_mask_softmax_sample(logits,  │ ← Kernel 3
                    │                                mask_i)│
                    │    token_i = sample()                 │
                    │    grammar.advance(token_i)           │
                    │    mask_{i+1} = grammar.get_mask()    │
                    │    density_{i+1} = popcount / V       │
                    │    if should_stop(density): break     │
                    └──────────┬───────────────────────────┘
                               │ K_adaptive draft tokens
                               ▼
                    ┌──────────────────────────────────────┐
                    │     Target Model (GPU)                │
                    │     single forward pass:              │
                    │                                       │
                    │  verify K+1 tokens                    │
                    │  grammar mask on all positions        │ ← Kernel 2
                    │  (vLLM already does this part)        │
                    └──────────────────────────────────────┘
```

---

## 4. Custom CUDA kernels

### Kernel 1: `popcount_density.cu` ⭐ (drives adaptive K)

**Purpose**: Count set bits in grammar bitmask → compute density ratio.

```cuda
// Input: bitmask (uint64_t array, ceil(vocab_size/64) elements)
// Output: float density = valid_tokens / vocab_size
// Latency target: < 1μs (vocab=151K → 2361 uint64s → 1 warp in ~8 iterations)

__global__ void popcount_density_kernel(
    const uint64_t* __restrict__ bitmask,
    int* total_count,
    int num_u64
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_count = 0;

    // Grid-stride loop with thread coarsening
    for (int i = tid; i < num_u64; i += gridDim.x * blockDim.x) {
        local_count += __popcll(bitmask[i]);  // Hardware popcount intrinsic
    }

    // Warp-level reduction via shuffle
    local_count += __shfl_down_sync(0xFFFFFFFF, local_count, 16);
    local_count += __shfl_down_sync(0xFFFFFFFF, local_count, 8);
    local_count += __shfl_down_sync(0xFFFFFFFF, local_count, 4);
    local_count += __shfl_down_sync(0xFFFFFFFF, local_count, 2);
    local_count += __shfl_down_sync(0xFFFFFFFF, local_count, 1);

    // Lane 0 of each warp atomic-adds to global counter
    if (threadIdx.x % 32 == 0) {
        atomicAdd(total_count, local_count);
    }
}
```

**HPC talking points**:
- `__popcll` is a single-instruction hardware intrinsic (not a loop)
- Warp shuffle reduction (`__shfl_down_sync`) avoids shared memory round-trip
- For vocab=151K: bitmask = 2361 × uint64 = ~19KB → fits in L2 cache
- One warp (32 threads) processes 2361 / 32 ≈ 74 elements/thread → ~74 iterations
- Total: launch + compute + reduce < 1μs on any modern GPU
- Memory coalescing: bitmask stored as contiguous uint64 array, consecutive threads access consecutive elements

**Profiling plan**: Use Nsight Compute to show:
- Occupancy (should be low utilization — this is latency-bound, not throughput-bound)
- Memory throughput (bitmask is tiny, fits in cache)
- Compare vs `thrust::count` or `cub::DeviceReduce` as baseline

### Kernel 2: `grammar_masked_acceptance.cu`

**Purpose**: Grammar-aware acceptance test in the verification path. For each draft token, compute `min(1, p_target_masked / p_draft_masked)` over the grammar-valid support, sample accept/reject, output first-reject index.

```cuda
// Fused: apply grammar mask to both target and draft logits,
// then compute acceptance probability and sample.
//
// Phase 1: Apply mask (set invalid logits to -inf)
// Phase 2: Compute masked softmax for both target and draft
// Phase 3: Compute acceptance ratio per token
// Phase 4: Sample accept/reject, find first rejection

__global__ void grammar_masked_acceptance_kernel(
    float* __restrict__ target_logits,    // [K+1, vocab_size]
    float* __restrict__ draft_logits,     // [K+1, vocab_size]
    const uint64_t* __restrict__ bitmask, // [K+1, ceil(V/64)]
    int* __restrict__ accept_mask,        // [K+1] output: 1=accept, 0=reject
    int* __restrict__ first_reject,       // scalar output
    int K_plus_1,
    int vocab_size,
    float threshold          // acceptance threshold (uniform sample compared against this)
) {
    // Each block handles one position (one of K+1)
    int pos = blockIdx.x;
    int tid = threadIdx.x;

    // Phase 1: Mask invalid tokens
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        int word = i / 64;
        int bit = i % 64;
        bool valid = (bitmask[pos * ceil_v_64 + word] >> bit) & 1;
        if (!valid) {
            target_logits[pos * vocab_size + i] = -INFINITY;
            draft_logits[pos * vocab_size + i] = -INFINITY;
        }
    }
    __syncthreads();

    // Phase 2-3: Reduction for max, sum, acceptance ratio
    // (details: standard numerically-stable softmax reduction in shared memory)
    // ...

    // Phase 4: Compare ratio vs threshold, atomic-min for first reject
}
```

**Correctness note**: When both draft and target are masked by the same grammar, the acceptance ratio `p_target_masked / p_draft_masked` operates over the **grammar-valid support only**. The renormalization must be consistent — this is the subtle correctness point that shows you understand the algorithm, not just copied it.

### Kernel 3: `fused_mask_softmax_sample.cu` (draft path)

**Purpose**: In the draft model's autoregressive loop, fuse three operations into one kernel: apply grammar mask → masked softmax → sample. Avoids three separate global memory reads of the vocab_size logits array.

```cuda
// Fused pipeline per draft step:
// 1. Read logits[i], check bitmask[i] → mask invalid to -inf
// 2. Online softmax: track running max + running sum (numerically stable)
// 3. Inverse CDF sampling: generate one token from masked distribution
//
// Memory reads: 1 pass over logits (vs 3 passes if unfused)
// Memory writes: 1 token output (vs full softmax vector if unfused)

__global__ void fused_mask_softmax_sample_kernel(
    const float* __restrict__ logits,
    const uint64_t* __restrict__ bitmask,
    int* sampled_token,
    float temperature,
    int vocab_size,
    uint32_t random_seed
) {
    // Shared memory for block-level reduction
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    float local_max = -INFINITY;
    float local_sum = 0.0f;

    // Phase 1: Mask + find max (online softmax step 1)
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        int word = i / 64;
        int bit = i % 64;
        bool valid = (bitmask[word] >> bit) & 1;
        float val = valid ? logits[i] / temperature : -INFINITY;
        local_max = fmaxf(local_max, val);
    }
    // Block-level max reduction in shared memory
    // ...

    // Phase 2: Compute exp(val - max) and sum
    // ...

    // Phase 3: Inverse CDF sampling
    // Generate random float r ∈ [0, sum), walk until cumulative > r
    // ...
}
```

**HPC talking points**:
- **Kernel fusion** is the canonical GPU optimization pattern: reduce global memory traffic
- For vocab=151K, logits array = 604KB (FP32) — reading it 3× (mask, softmax, sample) = 1.8MB. Fused = 604KB. 3× reduction in memory bandwidth.
- Online softmax algorithm (Milakov & Gimelshein, 2018): single-pass numerically stable softmax using running max + sum
- Temperature scaling fused into the first read — no extra pass
- Curand device API for in-kernel RNG

---

## 5. Repository structure

```
grammar-sd/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py                        ← builds CUDA extension (pybind11)
├── src/
│   ├── serve.py                    ← CLI: --schema X.json --prompt "..." → valid JSON
│   ├── speculator.py               ← orchestration: draft → adaptive_K → target → accept
│   ├── grammar.py                  ← thin wrapper around xgrammar (bitmask + state)
│   ├── adaptive_k.py               ← density → K mapping (threshold policy)
│   ├── configs.py                  ← 4 configs: free / verify-only / grammar-draft / grammarSD
│   ├── model_wrapper.py            ← HF model loading + KV cache mgmt
│   ├── baseline.py                 ← vanilla autoregressive (speedup reference)
│   ├── _kernels/
│   │   ├── popcount_density.cu     ← Kernel 1: grammar density computation
│   │   ├── grammar_acceptance.cu   ← Kernel 2: masked acceptance test
│   │   ├── fused_sample.cu         ← Kernel 3: fused mask+softmax+sample
│   │   ├── bindings.cpp            ← pybind11 bindings
│   │   └── kernel_loader.py
│   └── utils.py
├── benchmarks/
│   ├── bench.py                    ← 4 configs × schema complexity × K sweep
│   ├── schemas/                    ← person.json, toolcall.json, nested.json, JSONSchemaBench subset
│   ├── plots/                      ← acceptance chart, speedup chart, adaptive-K trace
│   └── results/
├── analysis/
│   ├── acceptance-gap.md           ← C1→C2: how grammar hurts spec decoding
│   ├── grammar-recovery.md         ← C2→C3: how grammar-guided drafting recovers
│   ├── adaptive-k-benefit.md       ← C3→C4: how adaptive K adds throughput
│   ├── failure-modes.md
│   └── correctness-note.md         ← masked-support renormalization + GAD/Φ scope
├── notes/                          ← breadth notes for interviews
│   ├── paged-attention.md
│   ├── async-scheduling.md
│   ├── sharding.md
│   ├── adaptive-spec-decoding.md   ← SpecDec++, AdaSpec, SGLang Adaptive survey
│   └── grammar-engines.md          ← xgrammar vs outlines vs llguidance
└── tests/
    ├── test_validity.py            ← 100% schema-valid output
    └── test_correctness.py         ← distribution matches target-only @ seed=0
```

---

## 6. Week-by-week milestones

### Week 1 (~12 hr): Spec decoding + grammar + configs 1-2

| Day | Hrs | Task | Deliverable |
|---|---|---|---|
| Mon | 2 | Cloud GPU setup, load Qwen2.5-1.5B + 0.5B, generate sample text | Both models respond |
| Tue | 2 | `baseline.py` vanilla AR; integrate xgrammar, emit valid JSON | Valid JSON, grammar works |
| Wed | 2 | `speculator.py` free spec decoding (C1), Python only | Free spec streams tokens |
| Thu | 2 | Add verify-only grammar mask (C2); observe acceptance drop | **The gap reproduced + measured** |
| Fri | 2 | KV rewind correctness; `test_validity.py` | All outputs schema-valid |
| Sat | 2 | First C1/C2 benchmark numbers | Acceptance gap quantified |
| Sun | — | Buffer / commit | Tagged `v0.1-gap-measured` |

**Week 1 success**: C1 & C2 working; acceptance gap reproduced and quantified.

### Week 2 (~14 hr): Grammar-guided draft (C3) + adaptive K (C4) + CUDA kernels

| Day | Hrs | Task | Deliverable |
|---|---|---|---|
| Mon | 2 | Implement C3 (grammar mask on draft); measure acceptance recovery | Recovery number exists |
| Tue | 2 | Implement `adaptive_k.py` (density → K threshold logic) in Python | C4 runs in Python |
| Wed | 2 | Measure C4: adaptive K effect on throughput | C4 beats C3 on throughput? |
| Thu | 2 | CUDA build setup (pybind11), Kernel 1: `popcount_density.cu` | Kernel imports, density matches Python |
| Fri | 3 | Kernel 2: `grammar_acceptance.cu`; validate vs Python reference | Acceptance matches |
| Sat | 3 | Kernel 3: `fused_mask_softmax_sample.cu`; wire all kernels | Full CUDA path runs |
| Sun | — | Buffer / commit | Tagged `v0.2-cuda-adaptive` |

**Week 2 success**: C3 and C4 working; all 3 CUDA kernels validated; C4 shows measurable throughput improvement.

### Week 3 (~12 hr): Benchmark sweep + analysis + README

| Day | Hrs | Task | Deliverable |
|---|---|---|---|
| Mon | 3 | `bench.py`: 4 configs × 3 schemas × K sweep | Sweep harness runs |
| Tue | 2 | Launch sweep; collect data | CSVs in results/ |
| Wed | 2 | Plots: acceptance recovery (C1-C4 bars), speedup, adaptive-K trace | PNGs in plots/ |
| Thu | 3 | Write `analysis/*.md` (3 ablation writeups) | Study writeups done |
| Fri | 2 | README: hook, money chart, adaptive-K story, kernels, related work | README done |

**Week 3 success**: Public repo with 4-config ablation study, adaptive K story front and center.

### Week 4 (buffer / B→C transition, ~6-10 hr)

| Day | Hrs | Task | Deliverable |
|---|---|---|---|
| Mon | 2 | Nsight profiling of all 3 kernels; profiling screenshots | `analysis/profiling.md` |
| Tue | 2 | Polish README, push public, update resume + battle map | Repo public |
| Wed-Fri | 2-6 | **B→C: Reproduce vLLM issue #34650**; draft fix; open PR | vLLM PR opened |

---

## 7. README structure

```markdown
# GrammarSD: Grammar-Aware Adaptive Speculative Decoding

Speculative decoding + structured output = **acceptance rate craters**.
The draft model doesn't know the grammar, wastes proposals on invalid tokens.

This project fixes it two ways:
1. **Grammar-guided drafting**: apply the grammar mask to the draft model too
2. **Adaptive speculation budget**: use grammar mask density to dynamically
   set K — speculate aggressively when constrained, conservatively when free

Result: acceptance recovers from 38% → 71%, throughput improves 1.3x → 2.8x+.

[Money chart: 4 bars — Free Spec / Verify-Only / Grammar-Guided / GrammarSD]

## Why grammar density is the right signal

[Table: existing adaptive K signals vs grammar density — forward-looking, zero-cost, per-position]

## Quick start
[3 lines: install, run with a schema, watch valid JSON stream]

## The four configs
[C1-C4 table + architecture diagram]

## Findings
- The gap: verify-only grammar drops acceptance to ~38%
- Grammar-guided drafting recovers to ~71%
- Adaptive K adds X% more throughput on top
- Effect scales with schema complexity

## Custom CUDA kernels
[popcount_density, grammar_acceptance, fused_sample — Nsight traces]

## Related work
[SpecDec++, SGLang Adaptive, GAD, Nie et al. — honest positioning]

## B→C: Upstream impact
[vLLM issues #27210, #34650 — this project's prototype is a PoC for fixes]
```

---

## 8. B→C transition plan

Phase B (this project) naturally feeds into Phase C (upstream contribution):

### Path 1: vLLM bug fix (lower risk)

- **Target**: Issue [#34650](https://github.com/vllm-project/vllm/issues/34650) — grammar mask silently not applied under spec decode + reasoning
- **Why B feeds C**: Your C3 implementation (grammar mask on draft path) is a working PoC showing how to correctly handle grammar in the spec decode loop
- **Effort**: 1-2 weeks (understand their architecture, adapt your fix, open PR)
- **Resume line**: "Fixed grammar mask application bug in vLLM speculative decoding (#34650), affecting structured output + reasoning workloads on B200 clusters"

### Path 2: SGLang adaptive-K extension (medium risk)

- **Target**: SGLang already has `AdaptiveController` with EMA-based K adaptation
- **Why B feeds C**: Your grammar-density signal can be added as an alternative policy alongside their EMA — your C4 benchmark proves the concept
- **Effort**: 2-3 weeks (integrate into their `SpecRuntimeState` infrastructure, benchmark on their test suite, open PR)
- **Resume line**: "Extended SGLang's adaptive speculative decoding with grammar-density-based per-position K scheduling, improving structured output throughput by X%"

### Path 3: vLLM grammar-mask-on-draft feature (higher risk, higher reward)

- **Target**: Add grammar mask to vLLM's draft proposer path (`llm_base_proposer.py`)
- **Why B feeds C**: Your entire C3+C4 implementation is the design doc + PoC
- **Effort**: 3-4 weeks (design discussion with maintainers, implement in their architecture, benchmark)
- **Resume line**: "Contributed grammar-aware adaptive speculative decoding to vLLM, the most-deployed LLM inference engine"

### Risk mitigation

- B is a **standalone deliverable** — even if C's PR is rejected or stalled, B has full value
- C's timeline is **reviewer-dependent** — don't let C block job search
- **Recommendation**: Start with Path 1 (bug fix) — fastest to merge, builds credibility for Path 2/3

---

## 9. Risk assessment

| Risk | Prob | Impact | Mitigation |
|---|---|---|---|
| Adaptive K (C4) shows minimal improvement over C3 | Medium | Reduces novelty | C3 (grammar-guided drafting) alone is still a valid contribution; C4 becomes "marginal improvement" rather than "major win" |
| Grammar density threshold needs heavy tuning | Low | Delays Week 2 | Start with simple 3-tier threshold; tune later |
| xgrammar API doesn't expose bitmask in useful format | Low | Blocks kernel integration | Fall back to `outlines` or manual JSON state machine |
| vLLM fixes #34650 before your PR | Low | Narrows Path 1 | Paths 2 and 3 still open; B is independent |
| Kernels too simple for HPC interviewers | Medium | Weakens HPC positioning | Kernel 3 (fused mask+softmax+sample) has real fusion depth; Nsight profiling shows optimization thinking |

---

## 10. Quick reference: key sources

### Adaptive speculative decoding
- SpecDec++ (ICML 2024 / COLM 2025): [arxiv.org/abs/2405.19715](https://arxiv.org/abs/2405.19715) 🟢 — threshold policy theorem, trained acceptance head
- SGLang Adaptive Spec: [sgl-project.github.io](https://sgl-project.github.io/advanced_features/adaptive_speculative_decoding.html) 🟢 — EMA-based, candidate_steps [1,3,7]
- AdaSpec: [arxiv.org/abs/2503.05096](https://arxiv.org/abs/2503.05096) 🟡 — SLO-driven, 66% speedup
- SpecRouter: [arxiv.org/abs/2505.07680](https://arxiv.org/abs/2505.07680) 🟡 — multi-level, token divergence routing
- TurboSpec: UC Berkeley PhD thesis, [EECS-2025-224](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-224.html) 🟡 — closed-loop control

### Grammar + structured output
- GAD (NeurIPS 2024): [arxiv.org/abs/2405.21047](https://arxiv.org/abs/2405.21047) 🟢 — local masking distributional bias
- Nie et al. (2026): [arxiv.org/abs/2605.07698](https://arxiv.org/abs/2605.07698) 🟡 — spec decoding + grammar, TV=0.996
- XGrammar: [github.com/mlc-ai/xgrammar](https://github.com/mlc-ai/xgrammar) 🟢
- JSONSchemaBench: [arxiv.org/abs/2501.10868](https://arxiv.org/abs/2501.10868) 🟢 — 10K real-world JSON schemas

### Production source code (verified 2026-06-26)
- vLLM `gpu_model_runner.py`: [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm/tree/main/vllm/v1) 🟢
- vLLM issue #27210: [EAGLE + structured output FSM crash](https://github.com/vllm-project/vllm/issues/27210) 🟢
- vLLM issue #34650: [Grammar mask not applied under spec decode + reasoning](https://github.com/vllm-project/vllm/issues/34650) 🟢
- SGLang `scheduler.py`: [github.com/sgl-project/sglang](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt) 🟢

---

## 11. Change log

- **2026-06-26 (v3)**: Major pivot from v2. Added adaptive K driven by grammar mask density as the primary novelty. Expanded from 3 configs to 4 (adding C4: GrammarSD full). Added competitive landscape analysis (SpecDec++, SGLang Adaptive, AdaSpec, SpecRouter). Restructured kernels (popcount_density as Kernel 1 ⭐). Added B→C transition plan with 3 upstream paths. Supersedes v2.
- **2026-06-24 (v2)**: Literature review integrated. Found GAD + Nie et al. Updated honesty boundary. Added vLLM/SGLang source code verification.
- **2026-06-23 (v1)**: Initial spec. Grammar-guided speculative decoding concept.
