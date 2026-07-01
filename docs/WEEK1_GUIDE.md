# Week 1: Grammar-Constrained Speculative Decoding (C1 + C2)

## Overview

Pure Python implementation (no custom CUDA kernels) of speculative decoding with grammar constraints using [xgrammar](https://github.com/mlc-ai/xgrammar).

**Configurations:**
- **C1 (Free Spec)** — Standard speculative decoding, no grammar constraints.
- **C2 (Verify-Only Grammar)** — Grammar bitmask applied only during target verification.

**Models:**
- Target: `Qwen/Qwen3.5-4B`
- Draft: `Qwen/Qwen3.5-0.8B`

---

## Google Colab Setup Guide

### Prerequisites

- Google account
- Basic Python knowledge
- ~15 GB GPU RAM (T4 works)

### Step 1: Open Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. **File → New notebook**
3. **Runtime → Change runtime type → T4 GPU → Save**

### Step 2: Clone Repo (Cell 1)

```python
!git clone https://github.com/shilonggrad-design/spec-decoding.git
%cd spec-decoding
```

### Step 3: Install Dependencies (Cell 2)

```python
!pip install torch transformers xgrammar accelerate
```

### Step 4: Mount Google Drive for Results (Cell 3)

```python
from google.colab import drive
drive.mount('/content/drive')
import os
os.makedirs('/content/drive/MyDrive/grammar-sd/results', exist_ok=True)
```

After running benchmarks, copy results to Drive:
```python
!cp -r results/ /content/drive/MyDrive/grammar-sd/
```

---

## Day-by-Day Instructions

### Day 1: Model Loading

```python
import sys
sys.path.insert(0, '/content/spec-decoding')
from src.baseline import load_models

target, draft, tokenizer = load_models()
print(f"Target: {target.config.vocab_size} vocab, {sum(p.numel() for p in target.parameters())/1e9:.1f}B params")
print(f"Draft:  {draft.config.vocab_size} vocab, {sum(p.numel() for p in draft.parameters())/1e9:.1f}B params")
```

Verify both models load and vocab sizes match.

### Day 2: Baseline (Vanilla AR + Grammar-Constrained)

```python
%cd /content/spec-decoding
!python src/baseline.py
```

- Check that grammar-constrained output is valid JSON
- Compare timing: vanilla AR vs grammar-constrained AR
- Note the average mask density

### Day 3: C1 — Free Speculative Decoding

```python
import sys
sys.path.insert(0, '/content/spec-decoding')
from src.speculator import load_models, speculative_decode

target, draft, tokenizer = load_models()
prompt_tokens = tokenizer.encode("Generate a person's information.", add_special_tokens=True)

result = speculative_decode(target, draft, tokenizer, prompt_tokens, K=5, config="C1", max_tokens=256)
print(f"Acceptance rate: {result['acceptance_rate']:.2%}")
print(f"Output: {result['text'][:300]}")
```

Expected: high acceptance rate (60–80%), faster than AR.

### Day 4: C2 — Verify-Only Grammar

```python
with open("schemas/schema_simple.json") as f:
    schema_str = f.read()

result = speculative_decode(target, draft, tokenizer, prompt_tokens,
                            K=5, config="C2", schema_str=schema_str, max_tokens=256)
print(f"Acceptance rate: {result['acceptance_rate']:.2%}")
print(f"Output: {result['text'][:300]}")
```

- Expect lower acceptance rate than C1 (draft is unconstrained but verification is strict)
- Output must be valid JSON matching the schema

### Day 5: Correctness Tests

```python
!cd /content/spec-decoding && python src/test_validity.py
```

All 5 tests should pass:
- `test_vocab_match` — vocab sizes match
- `test_ar_output_valid` — grammar AR produces valid JSON
- `test_spec_c1_matches_ar` — C1 ≈ vanilla AR (greedy)
- `test_spec_c2_output_valid` — C2 produces valid JSON
- `test_kv_rollback` — no stale KV cache after rejection

### Day 6: Full Benchmark

```python
!cd /content/spec-decoding && python src/bench_week1.py
```

This runs C1 and C2 across 3 schemas × 5 prompts = 30 trials. Results are saved to `results/week1_c1_c2.csv`.

Copy to Drive when done:
```python
!cp results/week1_c1_c2.csv /content/drive/MyDrive/grammar-sd/
```

---

## Keep-Alive (Prevent Session Timeout)

Add this cell and run it in the background:

```python
%%javascript
function ClickConnect() {
    console.log("Keeping alive...");
    document.querySelector("colab-connect-button").click();
}
setInterval(ClickConnect, 60000);
```

Or use a Python loop:
```python
import time
while True:
    time.sleep(300)
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| **T4 OOM** | Reduce `max_tokens` to 128, or add `torch_dtype=torch.float16` explicitly |
| **xgrammar import error** | `pip install --upgrade xgrammar` |
| **Session timeout** | Save results to Drive after every run (Step 4) |
| **Slow generation** | Make sure `device_map="auto"` is set; verify GPU with `!nvidia-smi` |
| **JSON parse failure** | The model may not have generated a complete JSON — increase `max_tokens` |
| **Acceptance rate = 0%** | Check that both models use the same tokenizer and same `torch_dtype` |

---

## Project Structure

```
spec-decoding/
├── schemas/
│   ├── schema_simple.json      # Person info (name, age, city, email)
│   ├── schema_tool_call.json   # Function call with enum
│   └── schema_nested.json      # Users array with nested addresses
├── src/
│   ├── __init__.py
│   ├── baseline.py             # Day 2: AR + grammar-constrained generation
│   ├── speculator.py           # Day 3-4: C1 + C2 speculative decoding
│   ├── bench_week1.py          # Day 6: Full benchmark
│   └── test_validity.py        # Day 5: Correctness tests
├── results/                    # Benchmark CSV output
└── WEEK1_GUIDE.md             # This file
```
