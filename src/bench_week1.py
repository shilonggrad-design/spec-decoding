#!/usr/bin/env python3
"""
bench_week1.py — Day 6: Benchmark C1 and C2 across schemas × prompts.

Runs speculative decoding (C1 and C2) for each combination of schema and prompt,
collects metrics, and saves results to a CSV file.
"""

from __future__ import annotations

import csv
import os
import sys
import time
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

CONFIGS = ["C1", "C2"]

K = 5  # speculation width
MAX_TOKENS = 256  # max tokens to generate per trial
OUTPUT_CSV = "results/week1_c1_c2.csv"

# CSV columns
CSV_FIELDS = [
    "config", "schema", "prompt_id", "acceptance_rate",
    "total_accepted", "total_drafted", "time_sec",
    "tokens_per_sec", "output_length", "rounds",
]


def load_schema(path: str) -> str:
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
    """Run a single benchmark trial. Returns a flat dict matching CSV_FIELDS."""
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    result = speculative_decode(
        target_model, draft_model, tokenizer, prompt_tokens,
        K=K, config=config,
        schema_str=schema_str if config == "C2" else None,
        max_tokens=MAX_TOKENS,
    )
    output_len = len(result["token_ids"])
    tps = output_len / max(result["time_sec"], 1e-9)

    return {
        "config": config,
        "schema": schema_str[:20] + "..." if schema_str else "none",
        "prompt_id": prompt[:40],
        "acceptance_rate": f"{result['acceptance_rate']:.4f}",
        "total_accepted": result["accepted_count"],
        "total_drafted": result["drafted_count"],
        "time_sec": f"{result['time_sec']:.3f}",
        "tokens_per_sec": f"{tps:.2f}",
        "output_length": output_len,
        "rounds": result.get("rounds", 0),
    }


def main() -> None:
    print("=" * 60)
    print("Week 1 — Day 6: Benchmark C1 + C2")
    print("=" * 60)

    base_dir = Path(__file__).resolve().parent.parent

    # Load models
    target_model, draft_model, tokenizer = load_models()
    print("[OK] Models loaded.")

    # Load schemas
    schemas_raw = {name: load_schema(base_dir / path) for name, path in SCHEMAS.items()}

    # Prepare output directory
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "week1_c1_c2.csv"

    rows: list[dict[str, str | int | float]] = []
    total_runs = len(CONFIGS) * len(SCHEMAS) * len(PROMPTS)
    run_idx = 0

    for config in CONFIGS:
        for schema_name, schema_str in schemas_raw.items():
            for prompt_idx, prompt in enumerate(PROMPTS):
                run_idx += 1
                print(
                    f"\n[{run_idx}/{total_runs}] "
                    f"config={config}  schema={schema_name}  prompt={prompt_idx}"
                )

                row = run_one(
                    target_model, draft_model, tokenizer,
                    prompt,
                    schema_str=schema_str,
                    config=config,
                )
                row["schema"] = schema_name
                row["prompt_id"] = str(prompt_idx)
                rows.append(row)

                print(f"  accept_rate={row['acceptance_rate']}  "
                      f"time={row['time_sec']}s  "
                      f"output_len={row['output_length']}")

    # ---- Write CSV ----
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[OK] Results saved to {csv_path}")

    # ---- Summary table ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Aggregate by (config, schema)
    from collections import defaultdict

    agg: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        agg[(row["config"], row["schema"])].append(row)

    header = f"{'Config':<6} {'Schema':<12} {'Avg Accept':>12} {'Avg TPS':>10} {'Avg Len':>10}"
    print(header)
    print("-" * len(header))

    for (config, schema), group in sorted(agg.items()):
        avg_accept = sum(float(r["acceptance_rate"]) for r in group) / len(group)
        avg_tps = sum(float(r["tokens_per_sec"]) for r in group) / len(group)
        avg_len = sum(r["output_length"] for r in group) / len(group)
        print(f"{config:<6} {schema:<12} {avg_accept:>12.2%} {avg_tps:>10.1f} {avg_len:>10.0f}")


if __name__ == "__main__":
    main()
