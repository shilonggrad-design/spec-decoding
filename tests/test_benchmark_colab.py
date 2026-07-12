"""
VeloSpec Comprehensive Benchmark — C3 (fixed K) vs C4 (adaptive K).

Tests adaptive K across diverse schemas to answer:
  "In what situations does adaptive K actually help?"

Usage in Colab:
  !git clone https://github.com/shilonggrad-design/spec-decoding
  %cd spec-decoding
  !pip install triton xgrammar transformers accelerate
  !python tests/test_benchmark_colab.py
"""

import os
import sys
import time
import json
import statistics

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from velospec import VeloSpec

TARGET_MODEL = "Qwen/Qwen3.5-4B"
DRAFT_MODEL = "Qwen/Qwen3.5-0.8B"

print("=" * 70)
print("VeloSpec — C3 vs C4 Comprehensive Benchmark")
print("=" * 70)
print(f"GPU: {torch.cuda.get_device_name(0)}")
print("=" * 70)

# ===========================================================================
# Test cases: designed to stress different density regimes
# ===========================================================================
TEST_CASES = [
    # --- Highly constrained (low density, adaptive K should win big) ---
    {
        "name": "enum_only",
        "desc": "Pure enum — extreme constraint",
        "prompt": "Pick a color from red, green, or blue.",
        "schema": {
            "type": "object",
            "properties": {
                "color": {"type": "string", "enum": ["red", "green", "blue"]}
            },
            "required": ["color"],
        },
        "max_tokens": 40,
    },
    {
        "name": "nested_enum",
        "desc": "Multiple enum fields — high structure",
        "prompt": "Create a server configuration.",
        "schema": {
            "type": "object",
            "properties": {
                "env": {"type": "string", "enum": ["dev", "staging", "prod"]},
                "region": {"type": "string", "enum": ["us-east", "us-west", "eu-central"]},
                "size": {"type": "string", "enum": ["small", "medium", "large"]},
            },
            "required": ["env", "region", "size"],
        },
        "max_tokens": 60,
    },
    # --- Mixed constraint (adaptive K should shine here) ---
    {
        "name": "mixed_short",
        "desc": "Short free text + structure",
        "prompt": "Generate a person profile.",
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        },
        "max_tokens": 50,
    },
    {
        "name": "mixed_medium",
        "desc": "Medium free text + multiple fields",
        "prompt": "Create a user account.",
        "schema": {
            "type": "object",
            "properties": {
                "username": {"type": "string"},
                "email": {"type": "string"},
                "role": {"type": "string", "enum": ["admin", "user", "guest"]},
                "active": {"type": "boolean"},
            },
            "required": ["username", "email", "role", "active"],
        },
        "max_tokens": 80,
    },
    # --- Free text heavy (high density, adaptive K → K=1, should be neutral) ---
    {
        "name": "free_text_heavy",
        "desc": "Long free text field — high density",
        "prompt": "Write a product review.",
        "schema": {
            "type": "object",
            "properties": {
                "product": {"type": "string"},
                "review": {"type": "string"},
                "rating": {"type": "integer"},
            },
            "required": ["product", "review", "rating"],
        },
        "max_tokens": 120,
    },
    {
        "name": "tool_call",
        "desc": "Function call — structure + short args",
        "prompt": "Search for the latest AI news.",
        "schema": {
            "type": "object",
            "properties": {
                "function": {"type": "string", "enum": ["search", "get", "post"]},
                "query": {"type": "string"},
            },
            "required": ["function", "query"],
        },
        "max_tokens": 40,
    },
]


def run_benchmark(config: str, K: int = 5):
    """Run all test cases for one config. Returns list of results."""
    backend = "triton"
    print(f"\n{'─' * 70}")
    print(f"Loading models for config={config}, K={K}, backend={backend}")
    print(f"{'─' * 70}")

    engine = VeloSpec(
        target_model=TARGET_MODEL,
        draft_model=DRAFT_MODEL,
        config=config,
        K=K,
        device="cuda",
        backend=backend,
    )

    results = []
    for tc in TEST_CASES:
        name = tc["name"]
        print(f"\n  [{config}] {name}: {tc['desc']}")

        try:
            result = engine.generate(
                prompt=tc["prompt"],
                schema=tc["schema"],
                max_tokens=tc["max_tokens"],
            )

            # Validate JSON output
            try:
                parsed = json.loads(result.text)
                valid = "✅"
            except:
                valid = "❌"

            # Per-round analysis for C4
            k_summary = ""
            if result.k_trace:
                ks = [k for _, k in result.k_trace]
                densities = [d for d, _ in result.k_trace]
                low_d = sum(1 for d in densities if d < 0.01)
                high_d = sum(1 for d in densities if d >= 0.01)
                k_summary = f"K_dist: low_density={low_d}rds, high_density={high_d}rds"

            print(f"    Output: {result.text[:80]}")
            print(f"    JSON valid: {valid}")
            print(f"    Accept: {result.acceptance_rate:.1%} | "
                  f"Tokens: {len(result.token_ids)} | "
                  f"Rounds: {result.rounds} | "
                  f"TPS: {result.tokens_per_sec:.1f}")
            if k_summary:
                print(f"    {k_summary}")
                print(f"    K values: {ks}")

            results.append({
                "name": name,
                "desc": tc["desc"],
                "acceptance": result.acceptance_rate,
                "tokens": len(result.token_ids),
                "rounds": result.rounds,
                "tps": result.tokens_per_sec,
                "valid_json": valid == "✅",
                "text": result.text,
                "k_trace": result.k_trace,
            })
        except Exception as e:
            print(f"    ❌ ERROR: {e}")
            results.append({
                "name": name, "desc": tc["desc"],
                "error": str(e),
                "acceptance": 0, "tokens": 0, "rounds": 0, "tps": 0,
                "valid_json": False, "text": "", "k_trace": [],
            })

    del engine
    torch.cuda.empty_cache()
    return results


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    # --- Run C3 (fixed K=5) ---
    print("\n" + "=" * 70)
    print("PHASE 1: C3 — Fixed K=5 (grammar-guided draft)")
    print("=" * 70)
    c3_results = run_benchmark("C3", K=5)

    # --- Run C4 (adaptive K, K_MAX=8) ---
    print("\n" + "=" * 70)
    print("PHASE 2: C4 — Adaptive K (grammar-guided draft + density-driven K)")
    print("=" * 70)
    c4_results = run_benchmark("C4", K=8)

    # =========================================================================
    # Comparison table
    # =========================================================================
    print("\n\n" + "=" * 70)
    print("COMPARISON: C3 (fixed K=5) vs C4 (adaptive K)")
    print("=" * 70)

    print(f"\n{'Test Case':<20s} | {'C3 accept':>10s} | {'C4 accept':>10s} | "
          f"{'C3 tps':>8s} | {'C4 tps':>8s} | {'C4/C3 tps':>10s} | {'C4 K dist':>20s}")
    print("-" * 105)

    c3_tps_list = []
    c4_tps_list = []
    c3_acc_list = []
    c4_acc_list = []

    for c3, c4 in zip(c3_results, c4_results):
        name = c3["name"]

        c3_acc = c3.get("acceptance", 0)
        c4_acc = c4.get("acceptance", 0)
        c3_t = c3.get("tps", 0)
        c4_t = c4.get("tps", 0)
        speedup = f"{c4_t / c3_t:.2f}×" if c3_t > 0 else "N/A"

        # K distribution summary
        if c4.get("k_trace"):
            ks = [k for _, k in c4["k_trace"]]
            k_counts = {}
            for k in ks:
                k_counts[k] = k_counts.get(k, 0) + 1
            k_str = ", ".join(f"K{k}={v}r" for k, v in sorted(k_counts.items()))
        else:
            k_str = "—"

        print(f"{name:<20s} | {c3_acc:>9.1%} | {c4_acc:>9.1%} | "
              f"{c3_t:>7.1f} | {c4_t:>7.1f} | {speedup:>10s} | {k_str:>20s}")

        if c3_t > 0:
            c3_tps_list.append(c3_t)
            c4_tps_list.append(c4_t)
            c3_acc_list.append(c3_acc)
            c4_acc_list.append(c4_acc)

    # Aggregate stats
    print("-" * 105)
    if c3_tps_list:
        avg_c3_tps = statistics.mean(c3_tps_list)
        avg_c4_tps = statistics.mean(c4_tps_list)
        avg_c3_acc = statistics.mean(c3_acc_list)
        avg_c4_acc = statistics.mean(c4_acc_list)
        avg_speedup = avg_c4_tps / avg_c3_tps if avg_c3_tps > 0 else 0

        print(f"\n{'AVERAGE':<20s} | {avg_c3_acc:>9.1%} | {avg_c4_acc:>9.1%} | "
              f"{avg_c3_tps:>7.1f} | {avg_c4_tps:>7.1f} | {avg_speedup:>9.2f}× |")

        print(f"\n📊 Summary:")
        print(f"   C3 avg throughput: {avg_c3_tps:.1f} tok/s")
        print(f"   C4 avg throughput: {avg_c4_tps:.1f} tok/s")
        print(f"   C4/C3 speedup:     {avg_speedup:.2f}×")
        print(f"   C3 avg acceptance: {avg_c3_acc:.1%}")
        print(f"   C4 avg acceptance: {avg_c4_acc:.1%}")

        # Per-case analysis
        print(f"\n📋 Per-case analysis:")
        for c3, c4 in zip(c3_results, c4_results):
            name = c3["name"]
            c3_t = c3.get("tps", 0)
            c4_t = c4.get("tps", 0)
            ratio = c4_t / c3_t if c3_t > 0 else 0

            if c4.get("k_trace"):
                densities = [d for d, _ in c4["k_trace"]]
                low_d_pct = sum(1 for d in densities if d < 0.01) / len(densities) * 100
            else:
                low_d_pct = 0

            verdict = ""
            if ratio > 1.2:
                verdict = "🟢 C4 wins"
            elif ratio < 0.9:
                verdict = "🔴 C3 wins"
            else:
                verdict = "🟡 Neutral"

            print(f"   {name:<20s}: C4/C3={ratio:.2f}× | "
                  f"low_density_rounds={low_d_pct:.0f}% | {verdict}")

    print("\n" + "=" * 70)
    print("Benchmark complete.")
    print("=" * 70)
