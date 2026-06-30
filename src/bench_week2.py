#!/usr/bin/env python3
"""
bench_week2.py — Week 2 Benchmark: Full C1-C4 ablation across schemas.

Runs all 4 configurations (C1, C2, C3, C4) for each combination of schema
and prompt, collects metrics, and saves results to CSV.

Configurations:
  C1: Free speculative decoding (no grammar)
  C2: Verify-only grammar (grammar on target only)
  C3: Grammar-guided draft (grammar on both draft + target)
  C4: GrammarSD full (C3 + adaptive K driven by density)

Usage:
  python src/bench_week2.py [--configs C1,C2,C3,C4] [--schemas simple,tool_call]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.speculator import load_models, speculative_decode  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROMPTS = [
    "Generate a person's information.",
    "Create a contact card for a software engineer named Alice who is 30 years old.",
    "Return a JSON object with user details.",
    "Provide structured data about a person from Seattle.",
    "Generate a profile for a machine learning engineer.",
]

SCHEMAS = {
    "simple": "schemas/schema_simple.json",
    "tool_call": "schemas/schema_tool_call.json",
    "nested": "schemas/schema_nested.json",
}

ALL_CONFIGS = ["C1", "C2", "C3", "C4"]

K = 5          # speculation width (for C1-C3; C4 uses adaptive K)
MAX_TOKENS = 256

CSV_FIELDS = [
    "config", "schema", "prompt_id",
    "acceptance_rate", "total_accepted", "total_drafted",
    "time_sec", "tokens_per_sec", "output_length", "rounds",
    "avg_density", "avg_k",  # C4-specific
]


def load_schema(path: str | Path) -> str:
    with open(path) as f:
        return f.read()


def run_one(
    target_model: Any,
    draft_model: Any,
    tokenizer: Any,
    prompt: str,
    schema_str: str | None,
    config: str,
) -> dict[str, Any]:
    """Run a single benchmark trial."""
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    result = speculative_decode(
        target_model, draft_model, tokenizer, prompt_tokens,
        K=K, config=config,
        schema_str=schema_str if config != "C1" else None,
        max_tokens=MAX_TOKENS,
    )
    output_len = len(result["token_ids"])
    tps = output_len / max(result["time_sec"], 1e-9)

    row: dict[str, Any] = {
        "config": config,
        "schema": "",
        "prompt_id": "",
        "acceptance_rate": f"{result['acceptance_rate']:.4f}",
        "total_accepted": result["accepted_count"],
        "total_drafted": result["drafted_count"],
        "time_sec": f"{result['time_sec']:.3f}",
        "tokens_per_sec": f"{tps:.2f}",
        "output_length": output_len,
        "rounds": result.get("rounds", 0),
        "avg_density": "",
        "avg_k": "",
    }

    # C4-specific metrics
    if "k_trace" in result and result["k_trace"]:
        avg_d = sum(d for d, _ in result["k_trace"]) / len(result["k_trace"])
        avg_k = sum(k for _, k in result["k_trace"]) / len(result["k_trace"])
        row["avg_density"] = f"{avg_d:.6f}"
        row["avg_k"] = f"{avg_k:.1f}"

    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 2 Benchmark: C1-C4 ablation")
    parser.add_argument("--configs", default="C1,C2,C3,C4",
                        help="Comma-separated configs to run (default: all)")
    parser.add_argument("--schemas", default="simple,tool_call,nested",
                        help="Comma-separated schemas to run")
    args = parser.parse_args()

    configs = args.configs.split(",")
    schema_names = args.schemas.split(",")

    print("=" * 60)
    print(f"Week 2 Benchmark: {' + '.join(configs)}")
    print("=" * 60)

    base_dir = Path(__file__).resolve().parent.parent

    # Load models
    target_model, draft_model, tokenizer = load_models()
    print("[OK] Models loaded.")

    # Load schemas
    schemas_raw: dict[str, str] = {}
    for name in schema_names:
        schemas_raw[name] = load_schema(base_dir / SCHEMAS[name])

    # Prepare output
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "week2_c1_c4.csv"

    rows: list[dict[str, Any]] = []
    total_runs = len(configs) * len(schema_names) * len(PROMPTS)
    run_idx = 0

    for config in configs:
        for schema_name in schema_names:
            for prompt_idx, prompt in enumerate(PROMPTS):
                run_idx += 1
                print(f"\n[{run_idx}/{total_runs}] config={config}  schema={schema_name}  prompt={prompt_idx}")

                row = run_one(
                    target_model, draft_model, tokenizer,
                    prompt,
                    schema_str=schemas_raw[schema_name],
                    config=config,
                )
                row["schema"] = schema_name
                row["prompt_id"] = str(prompt_idx)
                rows.append(row)

                extra = ""
                if row["avg_k"]:
                    extra = f"  avg_k={row['avg_k']}  density={row['avg_density']}"
                print(f"  accept_rate={row['acceptance_rate']}  "
                      f"time={row['time_sec']}s  output_len={row['output_length']}{extra}")

    # ---- Write CSV ----
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[OK] Results saved to {csv_path}")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    agg: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        agg[(row["config"], row["schema"])].append(row)

    header = f"{'Config':<6} {'Schema':<12} {'Avg Accept':>12} {'Avg TPS':>10} {'Avg Len':>10} {'Avg K':>8}"
    print(header)
    print("-" * len(header))

    for (config, schema), group in sorted(agg.items()):
        avg_accept = sum(float(r["acceptance_rate"]) for r in group) / len(group)
        avg_tps = sum(float(r["tokens_per_sec"]) for r in group) / len(group)
        avg_len = sum(r["output_length"] for r in group) / len(group)
        avg_k = ""
        if group[0]["avg_k"]:
            avg_k = f"{float(group[0]['avg_k']):.1f}"
        print(f"{config:<6} {schema:<12} {avg_accept:>12.2%} {avg_tps:>10.1f} {avg_len:>10.0f} {avg_k:>8}")

    # ---- Gap analysis ----
    if len(configs) >= 3 and "C1" in configs and "C2" in configs:
        print("\n--- Gap Analysis ---")
        for schema_name in schema_names:
            c1_rows = agg.get(("C1", schema_name), [])
            c2_rows = agg.get(("C2", schema_name), [])
            if c1_rows and c2_rows:
                c1_accept = sum(float(r["acceptance_rate"]) for r in c1_rows) / len(c1_rows)
                c2_accept = sum(float(r["acceptance_rate"]) for r in c2_rows) / len(c2_rows)
                gap = c1_accept - c2_accept
                print(f"  {schema_name}: C1={c1_accept:.2%} → C2={c2_accept:.2%}  gap={gap:+.2%}")

            if "C3" in configs:
                c3_rows = agg.get(("C3", schema_name), [])
                if c2_rows and c3_rows:
                    c2_accept = sum(float(r["acceptance_rate"]) for r in c2_rows) / len(c2_rows)
                    c3_accept = sum(float(r["acceptance_rate"]) for r in c3_rows) / len(c3_rows)
                    recovery = c3_accept - c2_accept
                    print(f"  {schema_name}: C2={c2_accept:.2%} → C3={c3_accept:.2%}  recovery={recovery:+.2%}")


if __name__ == "__main__":
    main()
