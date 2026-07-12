"""
VeloSpec End-to-End Inference Test — Colab A100.

This script loads real Qwen3.5 models and runs the full speculative decoding
pipeline with grammar constraints. It validates the entire system end-to-end.

Usage in Colab:
  !git clone https://github.com/shilonggrad-design/spec-decoding
  %cd spec-decoding
  !pip install triton xgrammar transformers accelerate
  !python tests/test_e2e_colab.py
"""

import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from velospec import VeloSpec

print("=" * 60)
print("VeloSpec — End-to-End Inference Test")
print("=" * 60)
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("=" * 60)

# ===========================================================================
# Models
# ===========================================================================
TARGET_MODEL = "Qwen/Qwen3.5-4B"
DRAFT_MODEL = "Qwen/Qwen3.5-0.8B"

# ===========================================================================
# Test schemas (simple → complex)
# ===========================================================================
SCHEMA_SIMPLE = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
    },
    "required": ["name", "age"],
}

SCHEMA_TOOL_CALL = {
    "type": "object",
    "properties": {
        "function": {"type": "string", "enum": ["search", "get", "post"]},
        "query": {"type": "string"},
    },
    "required": ["function", "query"],
}

# ===========================================================================
# Test prompts
# ===========================================================================
PROMPTS = [
    ("Generate a person profile", SCHEMA_SIMPLE),
    ("Search for AI news", SCHEMA_TOOL_CALL),
]


def run_config(config: str, K: int = 5, max_tokens: int = 64):
    """Run one full config and print results."""
    backend = "triton" if config != "C1" else "cuda"  # C1 doesn't need grammar

    print(f"\n{'─' * 50}")
    print(f"Config: {config} | K={K} | backend={backend}")
    print(f"{'─' * 50}")

    engine = VeloSpec(
        target_model=TARGET_MODEL,
        draft_model=DRAFT_MODEL,
        config=config,
        K=K,
        device="cuda",
        backend=backend,
    )

    results = []
    for prompt_text, schema in PROMPTS:
        print(f"\n  Prompt: \"{prompt_text}\"")
        try:
            t0 = time.perf_counter()
            result = engine.generate(
                prompt=prompt_text,
                schema=schema if config != "C1" else None,
                max_tokens=max_tokens,
            )
            wall = time.perf_counter() - t0

            print(f"  Output:    {result.text[:120]}")
            print(f"  Acceptance: {result.acceptance_rate:.1%}")
            print(f"  Tokens:     {len(result.token_ids)}")
            print(f"  Rounds:     {result.rounds}")
            print(f"  Throughput: {result.tokens_per_sec:.1f} tok/s")
            print(f"  Wall time:  {wall:.2f}s")
            if result.k_trace:
                densities = [d for d, _ in result.k_trace[:5]]
                ks = [k for _, k in result.k_trace[:5]]
                print(f"  K trace:    densities={['%.4f' % d for d in densities]}")
                print(f"             K values={ks}")

            results.append({
                "prompt": prompt_text,
                "acceptance": result.acceptance_rate,
                "tokens": len(result.token_ids),
                "rounds": result.rounds,
                "tps": result.tokens_per_sec,
                "text": result.text,
            })
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({"prompt": prompt_text, "error": str(e)})

    # Free memory before next config
    del engine
    torch.cuda.empty_cache()

    return results


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    all_results = {}

    # Run C1 first (no grammar, baseline)
    print("\n\n" + "=" * 60)
    print("Phase 1: C1 — Free Spec Decoding (no grammar)")
    print("=" * 60)
    all_results["C1"] = run_config("C1", K=5, max_tokens=64)

    # Run C3 (grammar-guided draft)
    print("\n\n" + "=" * 60)
    print("Phase 2: C3 — Grammar-Guided Draft")
    print("=" * 60)
    all_results["C3"] = run_config("C3", K=5, max_tokens=64)

    # Run C4 (adaptive K)
    print("\n\n" + "=" * 60)
    print("Phase 3: C4 — Adaptive K + Grammar-Guided Draft")
    print("=" * 60)
    all_results["C4"] = run_config("C4", K=8, max_tokens=64)

    # Summary comparison
    print("\n\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for config in ["C1", "C3", "C4"]:
        if config not in all_results:
            continue
        for r in all_results[config]:
            if "error" in r:
                print(f"  {config} | {r['prompt']:>30s} | ❌ {r['error'][:50]}")
            else:
                print(f"  {config} | {r['prompt']:>30s} | "
                      f"accept={r['acceptance']:.1%}  "
                      f"tps={r['tps']:.1f}  "
                      f"rounds={r['rounds']}")

    print("\n" + "=" * 60)
    print("E2E test complete.")
    print("=" * 60)
