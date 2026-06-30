# Week 2: Grammar-Guided Draft (C3) + Adaptive K (C4) + CUDA Kernels

## Overview

Week 2 has two halves:
- **Days 1-3 (Python):** C3 (grammar on draft) + C4 (adaptive K) — pure Python, same Colab setup as Week 1
- **Days 4-6 (CUDA):** 3 custom CUDA kernels — needs `nvcc` compiler (Colab A100 has it)

**Configurations:**
- **C3 (Grammar-Guided Draft)** — Grammar mask on BOTH draft and target
- **C4 (GrammarSD Full)** — C3 + adaptive K driven by grammar mask density

**Models:** Qwen3.5-4B (target) + Qwen3.5-0.8B (draft), vocab_size=248,320

---

## Colab Setup (same as Week 1)

```python
# Cell 1: Clone + pull latest
!git clone https://github.com/shilonggrad-design/spec-decoding.git
%cd spec-decoding
# If already cloned: !git pull

# Cell 2: Install deps
!pip install torch transformers xgrammar accelerate

# Cell 3: Mount Drive
from google.colab import drive
drive.mount('/content/drive')
import os
os.makedirs('/content/drive/MyDrive/grammar-sd/results', exist_ok=True)
```

---

## Day-by-Day Instructions

### Day 1 (Mon): C3 Quick Test — Acceptance Recovery

**Goal:** Verify C3 works, measure acceptance recovery on tool_call schema.

```python
# Cell: Run C1 vs C2 vs C3 on tool_call (the worst-gap schema)
import sys
sys.path.insert(0, '/content/spec-decoding')
from src.speculator import load_models, speculative_decode

target, draft, tokenizer = load_models()

with open("schemas/schema_tool_call.json") as f:
    schema_str = f.read()

prompt_tokens = tokenizer.encode("Call a function to search for AI news.", add_special_tokens=True)

results = {}
for cfg in ("C1", "C2", "C3"):
    result = speculative_decode(
        target, draft, tokenizer, prompt_tokens,
        K=5, config=cfg,
        schema_str=schema_str if cfg != "C1" else None,
        max_tokens=256,
    )
    results[cfg] = result
    print(f"{cfg}: accept={result['acceptance_rate']:.2%}  tokens={len(result['token_ids'])}  time={result['time_sec']:.1f}s")
    print(f"  Output: {result['text'][:200]}")
    print()
```

**Expected:**
- C1: ~77% accept (baseline, no grammar)
- C2: ~51% accept (the gap — draft unconstrained, target rejects)
- C3: **should recover to 70-85%** (draft now generates grammar-valid tokens)

**What to look for:**
1. C3 accept rate >> C2 accept rate on tool_call → **gap recovered**
2. C3 output is valid JSON (same as C2)
3. C3 output length similar to C2 (grammar early-stop still works)

**If C3 accept < C2:** Check that `matcher.rollback()` is called correctly. Draft advances matcher K steps, rollback undoes them so verify phase starts clean.

**Save results:**
```python
!cp -r results/ /content/drive/MyDrive/grammar-sd/
```

---

### Day 2 (Tue): C4 — Adaptive K Implementation

**Goal:** Implement `adaptive_k.py` — density → K mapping. Wire into speculator as C4.

#### Step 1: Create adaptive_k.py

```python
# src/adaptive_k.py
"""
Adaptive K controller driven by grammar mask density.

Low density (few valid tokens) → draft accuracy high → speculate aggressively (large K)
High density (many valid tokens) → draft accuracy low → speculate conservatively (small K)

Based on SpecDec++ (ICML 2024) threshold policy.
"""

def compute_density(bitmask_row, vocab_size):
    """Count valid tokens in bitmask / vocab_size."""
    valid = 0
    for word in bitmask_row:
        bits = word.item()
        if bits < 0:
            bits += 1 << 32
        valid += bin(bits).count("1")
    return valid / vocab_size

def adaptive_K(density, K_min=1, K_max=8, density_threshold=0.005):
    """
    Map grammar mask density to speculation width K.

    Thresholds tuned for Qwen3.5 (vocab=248,320):
    - density < 0.005 (fewer than ~1241 valid tokens) → K=K_max
    - density < 0.02  (fewer than ~4966 valid tokens) → K=(K_min+K_max)//2
    - else            → K=K_min
    """
    if density < density_threshold:
        return K_max
    elif density < density_threshold * 4:
        return (K_min + K_max) // 2
    else:
        return K_min
```

#### Step 2: Add C4 to speculator.py

C4 = C3 + adaptive K. The key change: instead of fixed K, compute K from
density at the START of each round.

Add to `speculator.py`:

```python
# In the Config type:
Config = Literal["C1", "C2", "C3", "C4"]

# In speculative_decode(), after building grammar pipeline, before the main loop:
from src.adaptive_k import compute_density, adaptive_K

# Inside the while loop, before draft phase:
if config == "C4" and matcher is not None and bitmask is not None:
    # Peek at current position density to decide K
    need_apply = matcher.fill_next_token_bitmask(bitmask)
    if need_apply:
        density = compute_density(bitmask[0], vocab_size)
        current_K = adaptive_K(density)
    else:
        current_K = K  # fallback to default
    # Rollback the peek (we haven't accepted anything yet)
    # fill_next_token_bitmask doesn't advance state, so no rollback needed
else:
    current_K = K
```

Then use `current_K` instead of `K` when calling `draft_k_tokens`.

#### Step 3: Track K decisions for analysis

```python
# Add to result dict:
if config == "C4":
    result["k_trace"] = k_trace  # list of (density, K) per round
```

---

### Day 3 (Wed): C4 Benchmark — Adaptive K Effect

**Goal:** Measure if C4's adaptive K beats C3's fixed K on throughput.

#### Step 1: Run C3 vs C4 on all 3 schemas

```python
import sys
sys.path.insert(0, '/content/spec-decoding')
from src.speculator import load_models, speculative_decode
import json

target, draft, tokenizer = load_models()

schemas = {
    "simple": open("schemas/schema_simple.json").read(),
    "tool_call": open("schemas/schema_tool_call.json").read(),
    "nested": open("schemas/schema_nested.json").read(),
}

prompts = [
    "Generate a person's information in JSON format.",
    "Create a user profile with name, age, and address.",
    "Return product details with nested specifications.",
    "Build a database record for an employee.",
    "Generate a configuration object for a web server.",
]

for schema_name, schema_str in schemas.items():
    print(f"\n{'='*60}")
    print(f"Schema: {schema_name}")
    print(f"{'='*60}")
    for cfg in ("C3", "C4"):
        for i, prompt in enumerate(prompts):
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
            result = speculative_decode(
                target, draft, tokenizer, prompt_tokens,
                K=5, config=cfg, schema_str=schema_str, max_tokens=256,
            )
            tps = len(result['token_ids']) / max(result['time_sec'], 0.001)
            print(f"  {cfg} prompt={i}: accept={result['acceptance_rate']:.2%}  "
                  f"tps={tps:.1f}  tokens={len(result['token_ids'])}")
```

#### Step 2: Update bench_week1.py → bench_week2.py

Add C3 and C4 configs to the benchmark harness.

**Expected:**
- C4 throughput > C3 throughput on schemas with high density variance
- C4 uses small K on constrained positions (saves wasted draft compute)
- C4 uses large K on free-form positions (more accepted tokens per round)

**Save results:**
```python
!cp results/week2_c3_c4.csv /content/drive/MyDrive/grammar-sd/
```

---

### Day 4 (Thu): CUDA Build Setup + Kernel 1 (popcount_density)

**Goal:** Set up CUDA compilation, write + validate the density kernel.

#### Step 0: Verify CUDA toolkit

```python
!nvcc --version   # Should show CUDA 12.x or 13.x
!nvidia-smi       # Should show A100
```

#### Step 1: Create kernel files

Create `src/_kernels/popcount_density.cu`:

```cuda
// Kernel 1: popcount_density
// Counts set bits in grammar bitmask → density = valid_tokens / vocab_size
// Latency target: < 1μs (vocab=248320 → 7761 int32s → ~243 uint64s)

#include <cuda_runtime.h>

__global__ void popcount_density_kernel(
    const uint32_t* __restrict__ bitmask,  // ceil(vocab_size/32) elements
    int* total_count,
    int num_words
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_count = 0;

    // Grid-stride loop
    for (int i = tid; i < num_words; i += gridDim.x * blockDim.x) {
        local_count += __popc(bitmask[i]);  // int32 popcount (not __popcll)
    }

    // Warp-level reduction via shuffle
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_count += __shfl_down_sync(0xFFFFFFFF, local_count, offset);
    }

    // Lane 0 of each warp atomic-adds
    if (threadIdx.x % 32 == 0) {
        atomicAdd(total_count, local_count);
    }
}
```

Create `src/_kernels/bindings.cpp`:

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

void popcount_density_launcher(
    const torch::Tensor& bitmask,
    torch::Tensor& total_count,
    int num_words
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("popcount_density", &popcount_density_launcher, "Popcount density kernel");
}
```

Create `setup.py`:

```python
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="grammar_sd_kernels",
    ext_modules=[
        CUDAExtension(
            name="grammar_sd_kernels",
            sources=[
                "src/_kernels/popcount_density.cu",
                "src/_kernels/bindings.cpp",
            ],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

#### Step 2: Build

```python
!python setup.py build_ext --inplace
```

#### Step 3: Validate vs Python

```python
import torch
import grammar_sd_kernels

# Create a test bitmask (7761 int32s, vocab_size=248320)
bitmask = torch.zeros(7761, dtype=torch.int32)
bitmask[0] = 0xFFFFFFFF  # 32 valid tokens in first word
bitmask[1] = 0x0000FFFF  # 16 valid tokens in second word
# Total valid = 48, density = 48 / 248320 = 0.000193

# Python reference
valid_py = 48
density_py = valid_py / 248320

# CUDA kernel
total_count = torch.zeros(1, dtype=torch.int32, device='cuda')
bitmask_gpu = bitmask.cuda()
grammar_sd_kernels.popcount_density(bitmask_gpu, total_count, 7761)
valid_cuda = total_count.item()
density_cuda = valid_cuda / 248320

print(f"Python:  valid={valid_py}, density={density_py:.8f}")
print(f"CUDA:    valid={valid_cuda}, density={density_cuda:.8f}")
print(f"Match: {valid_py == valid_cuda}")
```

**Must pass before moving on.** CUDA popcount must exactly match Python count.

---

### Day 5 (Fri): Kernel 2 (grammar_acceptance) — OPTIONAL

> ⚠️ This kernel is the most complex (multi-phase reduction). If time is tight,
> skip to Day 6 (Kernel 3) which is more interview-impressive.
> Kernel 2 can be replaced by existing xgrammar `apply_token_bitmask_inplace`.

**Goal:** Fused grammar-masked acceptance test for the verify phase.

Create `src/_kernels/grammar_acceptance.cu`:

```cuda
// Kernel 2: grammar_masked_acceptance
// Fuses: apply mask → softmax → accept/reject → find first rejection
// One block per position (K+1 positions), threads cooperatively process vocab

__global__ void grammar_masked_acceptance_kernel(
    float* __restrict__ target_logits,    // [K+1, vocab_size]
    const uint32_t* __restrict__ bitmask, // [K+1, ceil(V/32)]
    int* __restrict__ accept_mask,        // [K+1] output
    int K_plus_1,
    int vocab_size
) {
    int pos = blockIdx.x;
    int tid = threadIdx.x;

    // Phase 1: Mask invalid tokens → -inf, find max
    float local_max = -INFINITY;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        int word = i / 32;
        int bit = i % 32;
        bool valid = (bitmask[pos * (vocab_size/32 + 1) + word] >> bit) & 1;
        if (!valid) {
            target_logits[pos * vocab_size + i] = -INFINITY;
        } else {
            float val = target_logits[pos * vocab_size + i];
            local_max = fmaxf(local_max, val);
        }
    }

    // Phase 2: Block-level max reduction (shared memory)
    // ... (standard parallel reduction)

    // Phase 3: argmax over masked logits = target's preferred token
    // Compare against draft token → write accept/reject
}
```

**Validation:** Compare kernel output against `xgrammar.apply_token_bitmask_inplace` + argmax.

---

### Day 6 (Sat): Kernel 3 (fused_mask_softmax_sample) — DRAFT PATH

**Goal:** Fuse mask + softmax + sample into single kernel for draft phase.

This is the most interview-impressive kernel because:
- **Kernel fusion** = canonical GPU optimization pattern
- **Online softmax** = single-pass numerically stable algorithm
- **3× memory bandwidth reduction** (970KB read vs 2.9MB)

Create `src/_kernels/fused_sample.cu`:

```cuda
// Kernel 3: fused_mask_softmax_sample
// Fuses 3 ops into 1 kernel pass:
// 1. Read logits[i], check bitmask → mask invalid to -inf
// 2. Online softmax: running max + running sum (numerically stable)
// 3. Inverse CDF sampling → output 1 token
//
// Memory: 1 pass over logits (970KB) vs 3 passes unfused (2.9MB)

__global__ void fused_mask_softmax_sample_kernel(
    const float* __restrict__ logits,     // [vocab_size]
    const uint32_t* __restrict__ bitmask, // [ceil(V/32)]
    int* sampled_token,                   // [1] output
    float temperature,
    int vocab_size,
    unsigned int seed
) {
    // Phase 1: Mask + find max (online softmax step 1)
    // Phase 2: Compute exp(val - max) and sum
    // Phase 3: Inverse CDF sampling
    // (Implementation details left as exercise — see PROJECT_SPEC.md §4)
}
```

**Validation:** Output token must match Python reference (`apply_mask → softmax → argmax`).

---

### Day 7 (Sun): Buffer + Tag

```bash
# Commit all Week 2 work
git add -A
git commit -m "Week 2: C3 + C4 + CUDA kernels"
git tag -a v0.2-cuda-adaptive -m "C3 grammar-guided draft, C4 adaptive K, 3 CUDA kernels"

# Push
git push origin main --tags
```

---

## Expected Week 2 Results

| Config | tool_call Accept | tool_call Throughput | Key Insight |
|--------|:---:|:---:|---|
| C1 (free spec) | ~77% | baseline | No grammar |
| C2 (verify-only) | ~51% | -34% | **The gap** |
| C3 (grammar-guided) | **~75-85%** | **~baseline** | Gap recovered |
| C4 (adaptive K) | ~75-85% | **+10-20%** | Less wasted draft compute |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| **C3 accept < C2** | Check `matcher.rollback()` is called after draft phase |
| **C3 [WARN] Grammar rejected accepted token** | Matcher state desync — verify rollback count = len(draft_tokens) |
| **C4 slower than C3** | Adaptive K overhead may dominate for short outputs; check k_trace |
| **CUDA build fails** | `!pip install ninja`, check `!nvcc --version` |
| **Kernel result mismatch** | Check bitmask dtype (int32 not int64), popcount uses `__popc` not `__popcll` |
| **OOM on A100** | K_max=8 uses more draft compute; reduce to K_max=6 |

---

## Week 2 Success Criteria

- [x] C3 works, acceptance gap recovered on tool_call
- [x] C4 works, adaptive K traces show density-driven K changes
- [x] C4 throughput ≥ C3 throughput
- [x] Kernel 1 (popcount) validated vs Python
- [ ] Kernel 3 (fused sample) validated vs Python *(stretch)*
- [ ] Tagged `v0.2-cuda-adaptive`
