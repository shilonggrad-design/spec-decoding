# Week 1 — Day 2 Baseline Results

**GPU:** A100-SXM4-40GB (Colab Pro)
**Date:** 2026-06-29
**Models:** Qwen3.5-4B (target) + Qwen3.5-0.8B (draft), vocab_size=248320

## Baseline Comparison

| Config | Output Tokens | Time (s) | tok/s | Notes |
|--------|-------------|----------|-------|-------|
| Vanilla AR | 256 | 16.307 | 15.7 | No grammar |
| Grammar AR | 21 | 4.300 | 4.9 | schema_simple, terminated early (3 required fields met) |

## Key Metrics

- **Mask density (avg):** 0.2703 (27.0% of vocab valid per step)
- **GPU vs T4:** Vanilla AR ~18× faster on A100 (16.3s vs ~300s on T4)

## Notes

- Grammar AR terminated at 21 tokens because xgrammar detected schema completion + stop token
- Mask density 27% means strong constraint — good test case for C2 accept rate drop
- A100 40GB has plenty of headroom for both models (~10GB used out of 40GB)
