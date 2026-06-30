# Week 2: Grammar-Guided Draft (C3) + Adaptive K (C4) + CUDA Kernels

## Overview

Week 2 has two halves:
- **Days 1-3 (Python):** C3 (grammar on draft) + C4 (adaptive K) — pure Python
- **Days 4-6 (CUDA):** 3 custom CUDA kernels — needs `nvcc` compiler

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

## Day 1 (Mon): C3 Quick Test — Acceptance Recovery ✅ DONE

C3 code is already in `speculator.py`. Run:

```python
import sys
sys.path.insert(0, '/content/spec-decoding')
from src.speculator import load_models, speculative_decode

target, draft, tokenizer = load_models()

with open("schemas/schema_tool_call.json") as f:
    schema_str = f.read()

prompt_tokens = tokenizer.encode("Call a function to search for AI news.", add_special_tokens=True)

for cfg in ("C1", "C2", "C3"):
    result = speculative_decode(
        target, draft, tokenizer, prompt_tokens,
        K=5, config=cfg,
        schema_str=schema_str if cfg != "C1" else None,
        max_tokens=256,
    )
    print(f"{cfg}: accept={result['acceptance_rate']:.2%}  tokens={len(result['token_ids'])}  time={result['time_sec']:.1f}s")
    print(f"  Output: {result['text'][:200]}")
```

**Expected:**
- C1: ~89% accept (baseline, no grammar)
- C2: ~62% accept (the gap — draft unconstrained, target rejects)
- C3: **~100% accept** (draft now generates grammar-valid tokens)

---

## Day 2 (Tue): C4 — Adaptive K ✅ CODE DONE

C4 = C3 + density-driven K selection. Code is in `speculator.py` + `adaptive_k.py`.

```python
import sys
sys.path.insert(0, '/content/spec-decoding')
from src.speculator import load_models, speculative_decode

target, draft, tokenizer = load_models()

with open("schemas/schema_tool_call.json") as f:
    schema_str = f.read()

prompt_tokens = tokenizer.encode("Call a function to search for AI news.", add_special_tokens=True)

result = speculative_decode(
    target, draft, tokenizer, prompt_tokens,
    K=5, config="C4", schema_str=schema_str, max_tokens=256,
)

print(f"C4: accept={result['acceptance_rate']:.2%}")
print(f"Tokens: {len(result['token_ids'])}")
print(f"Time: {result['time_sec']:.3f}s")
print(f"K trace: {result.get('k_trace', 'N/A')}")
print(f"Output: {result['text'][:300]}")
```

**What k_trace shows:**
Each entry is `(density, K_chosen)`. You should see:
- Low density positions (JSON keys) → K=8 (speculate aggressively)
- High density positions (free text) → K=1 (conserve compute)

---

## Day 3 (Wed): C4 Full Benchmark

Run the Week 2 benchmark script:

```python
!cd /content/spec-decoding && git pull
!python src/bench_week2.py
```

This runs **C1/C2/C3/C4 × 3 schemas × 5 prompts = 60 trials**.

To run specific configs/schemas (faster):

```python
!python src/bench_week2.py --configs C3,C4 --schemas tool_call
```

Output: `results/week2_c1_c4.csv` + summary table with gap analysis.

**Save to Drive:**
```python
!cp results/week2_c1_c4.csv /content/drive/MyDrive/grammar-sd/
```

---

## Day 4 (Thu): CUDA Build Setup + Kernel 1 (popcount_density)

### Step 0: Verify CUDA toolkit

```python
!nvcc --version   # Should show CUDA 12.x or 13.x
!nvidia-smi       # Should show A100
```

### Step 1: Build the extension

```python
!cd /content/spec-decoding && git pull
!pip install ninja  # faster builds
!python setup.py build_ext --inplace
```

This compiles 3 kernels into `grammar_sd_kernels.so`:
1. `popcount_density` — count valid tokens in bitmask
2. `grammar_masked_argmax` — fused mask + argmax (verify path)
3. `fused_sample` — fused mask + softmax + sample (draft path)

### Step 2: Validate all kernels

```python
!python src/test_cuda_kernels.py
```

Expected output:
```
✅ Kernel 1 (popcount): CUDA=48 == Python=48
✅ Kernel 1 (sparse): CUDA=3 == Python=3
✅ Kernel 2 (argmax): CUDA=100 == Python=100
✅ Kernel 3 (greedy): CUDA=50 == Python=50
⚡ Popcount speedup: Python=X.Xms → CUDA=0.0XXms (N×)
Results: 5 passed, 0 failed
```

### Step 3: Measure kernel speedup

The test script includes a speedup measurement. Expected:
- Python popcount: ~2-5ms for 7761 int32 words
- CUDA popcount: ~0.01-0.1ms
- Speedup: 50-500×

---

## Day 5-6 (Fri-Sat): CUDA Kernel Deep Dive (OPTIONAL)

If you have time and want to go deeper:

### Kernel Architecture (all 3 kernels are ready to build)

| Kernel | File | Purpose | HPC Talking Point |
|--------|------|---------|-------------------|
| 1 | `popcount_density.cu` | Count valid tokens in bitmask | `__popc` intrinsic + warp shuffle reduction |
| 2 | `grammar_masked_argmax.cu` | Fused mask+argmax for K+1 positions | One block per position, parallel argmax |
| 3 | `fused_sample.cu` | Fused mask+softmax+sample | Online softmax, 5× memory reduction |

### Profiling with Nsight

```python
# If nsight is available on Colab
!pip install nvtx
# Then run kernel with profiling markers
```

### Kernel loader API

```python
from src._kernels.kernel_loader import (
    is_cuda_available,
    popcount_density,
    grammar_masked_argmax,
    fused_sample,
)

# Check if CUDA is built
print(is_cuda_available())  # True/False

# Kernel 1: count valid tokens
count, density = popcount_density(bitmask, vocab_size=248320)

# Kernel 2: masked argmax (single position)
token = grammar_masked_argmax(logits, bitmask, vocab_size=248320, num_words=7761)

# Kernel 3: fused sample (greedy)
token = fused_sample(logits, bitmask, vocab_size=248320, num_words=7761, temperature=0)
```

---

## Day 7 (Sun): Buffer + Tag

```bash
# Commit results
cd /content/spec-decoding
git add -A
git commit -m "Week 2: C3+C4 benchmark + CUDA kernels validated"

# Tag
git tag -a v0.2-cuda-adaptive -m "C3 grammar-guided draft, C4 adaptive K, 3 CUDA kernels"

# Push
git push origin main --tags
```

---

## File Structure (Week 2)

```
spec-decoding/
├── setup.py                          ← CUDA build (NEW)
├── src/
│   ├── speculator.py                 ← C1/C2/C3/C4 (UPDATED)
│   ├── adaptive_k.py                 ← density → K controller (NEW)
│   ├── bench_week2.py                ← C1-C4 benchmark (NEW)
│   ├── test_cuda_kernels.py          ← kernel validation (NEW)
│   └── _kernels/                     ← CUDA kernels (NEW)
│       ├── __init__.py
│       ├── popcount_density.cu       ← Kernel 1
│       ├── grammar_masked_argmax.cu  ← Kernel 2
│       ├── fused_sample.cu           ← Kernel 3
│       ├── bindings.cpp              ← pybind11 bindings
│       └── kernel_loader.py          ← Python wrapper with fallback
├── results/
│   ├── day6_benchmark.md             ← Week 1 results
│   └── week2_c1_c4.csv               ← Week 2 results (generated)
└── WEEK2_GUIDE.md                    ← This file
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| **C4 ImportError for adaptive_k** | Already fixed with 3-level fallback import in speculator.py |
| **CUDA build fails: nvcc not found** | Colab A100 has nvcc pre-installed. Run `!nvcc --version` to verify |
| **CUDA build fails: torch not found** | `!pip install torch` then retry `!python setup.py build_ext --inplace` |
| **Kernel result mismatch** | Check bitmask dtype is `int32` not `int64`. popcount uses `__popc` (32-bit) not `__popcll` (64-bit) |
| **grammar_sd_kernels import error** | After build, the `.so` file is in repo root. Run from repo root: `cd /content/spec-decoding` |
| **C4 slower than C3** | Adaptive K overhead may dominate for short outputs. Check `k_trace` — if all K=1, density is too high |
| **OOM on A100** | K_max=8 uses more draft compute per round. Reduce in `adaptive_k.py`: `K_MAX = 6` |

---

## Week 2 Success Criteria

- [ ] C3 works, acceptance gap recovered on tool_call (~100%)
- [ ] C4 works, k_trace shows density-driven K changes
- [ ] C4 throughput ≥ C3 throughput
- [ ] CUDA extension builds successfully
- [ ] All 5 kernel validation tests pass
- [ ] Tagged `v0.2-cuda-adaptive`
