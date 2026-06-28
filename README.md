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

## Related

- SpecDec++ (ICML 2024) — adaptive candidate length via trained acceptance head
- SGLang Adaptive Spec — EMA-based K adaptation (reactive)
- GAD (NeurIPS 2024) — local masking distributional bias
- Nie et al. (2026) — spec decoding + grammar distributional impossibility

## License

MIT
