<div align="center">
<img src="assets/logo.png" alt="VeloSpec" height="100"/>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-custom%20kernels-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Triton](https://img.shields.io/badge/Triton-fused%20kernel-orange.svg)](https://triton-lang.org/)
</div>

**VeloSpec** speeds up structured LLM output (JSON, tool calls) by combining
grammar-aware speculative decoding with an adaptive speculation budget —
4 custom GPU kernels (3 CUDA + 1 Triton), benchmarked end-to-end on A100.

---

## Why

Speculative decoding accelerates LLM inference by having a small draft model
guess K tokens ahead, then verifying them in one target forward pass. But when
generating **structured output** (JSON, tool calls), the draft model doesn't know
the grammar constraints and wastes proposals on invalid tokens — acceptance
rate craters, speedup evaporates.

## What VeloSpec Does

| Technique | Effect |
|-----------|--------|
| **Grammar-guided draft** — applies the grammar bitmask to the draft model too | Draft only proposes valid tokens → acceptance stays high |
| **Adaptive speculation budget** — reads grammar mask density to set K dynamically | Speculate aggressively when constrained (K=8), conserve when free (K=1) |

No training required. The density signal is free — it's a popcount on the
existing grammar bitmask.

## Results

Benchmarked on Qwen3.5-4B / 0.8B (vocab=248,320), end-to-end on A100-SXM4.

### Throughput: adaptive K vs fixed K

| Schema type | C3 (fixed K) accept | C4 (adaptive K) accept | C4/C3 throughput |
|-------------|:---:|:---:|:---:|
| Pure enum | 80% | 80% | **2.73×** |
| Nested enum (3 fields) | 100% | 100% | **1.64×** |
| Enum + short text | 85% | 92% | **1.62×** |
| Enum + text (mixed) | 95% | 95% | 1.15× |
| Long free text | 79% | 80% | 1.14× |
| Short free text | 92% | 85% | 0.96× |

**Average: 1.35× throughput, comparable acceptance (88.5% vs 88.3%).**

Adaptive K wins big on structured schemas and correctly backs off on free text.

### Triton fused logit processor

Single kernel replaces xgrammar mask + PyTorch argmax + CPU popcount.
Batch mode processes all K+1 verify positions in one launch:

| K+1 positions | Triton | PyTorch baseline | Speedup |
|:---:|:---:|:---:|:---:|
| 1 | 77 μs | 101 μs | 1.3× |
| 6 (K=5) | 83 μs | 649 μs | **7.8×** |
| 8 | 76 μs | 877 μs | 11.5× |
| 12 | 77 μs | 1331 μs | 17.3× |

Batch time stays flat as K grows — A100's 108 SMs parallelize rows for free.

### CUDA kernels

| Kernel | Technique | Speedup |
|--------|-----------|---------|
| `popcount_density.cu` | `__popc` + warp shuffle reduction | **111×** |
| `grammar_masked_argmax.cu` | Fused mask+argmax, 1 block/position | — |
| `fused_sample.cu` | Online softmax, 5× memory reduction | — |

## Quickstart

```bash
pip install velospec
```

```python
from velospec import VeloSpec

engine = VeloSpec(
    target_model="Qwen/Qwen3.5-4B",
    draft_model="Qwen/Qwen3.5-0.8B",
    config="C4",  # Grammar-guided draft + adaptive K
)

result = engine.generate(
    prompt="Call a function to search for AI news.",
    schema={
        "type": "object",
        "properties": {
            "function": {"type": "string", "enum": ["search", "get", "post"]},
            "arguments": {"type": "object"},
        },
    },
)

print(result.text)            # {"function": "search", ...}
print(result.acceptance_rate)  # 0.92
print(result.tokens_per_sec)   # 5.3
```

## How Adaptive K Works

The grammar bitmask density varies by position — it's a free forward-looking
signal for how constrained the next token is:

```
Position:     {"name": "___    ←  only ~2 tokens valid → density 0.001% → K=8
Position:     "value": "hello  ←  ~5000+ tokens valid → density 2%   → K=1
```

VeloSpec reads the bitmask popcount before each draft round and sets K:

```
density < 0.005 → K=8    (high constraint, draft can't miss)
density < 0.02  → K=4    (moderate)
else            → K=1    (low constraint, draft will likely diverge)
```

This is a **zero-cost, deterministic** signal — no trained prediction head,
no EMA lag. [SpecDec++ (ICML 2024)](https://arxiv.org/abs/2405.18466) proved
that the optimal K policy is a threshold policy; grammar density is a
deterministic estimator of that threshold.

## License

MIT
