#!/usr/bin/env python3
"""
cli.py — VeloSpec command-line interface.

Usage:
    # Generate structured output
    velospec generate \
        --target Qwen/Qwen3.5-4B \
        --draft Qwen/Qwen3.5-0.8B \
        --config C4 \
        --schema schemas/tool_call.json \
        "Call a function to search for AI news"

    # Benchmark
    velospec bench \
        --target Qwen/Qwen3.5-4B \
        --draft Qwen/Qwen3.5-0.8B \
        --configs C3,C4 \
        --schemas tool_call
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from velospec import VeloSpec

# ---------------------------------------------------------------------------
# Default prompts and schemas for benchmarking
# ---------------------------------------------------------------------------
DEFAULT_PROMPTS = [
    "Generate a person's information.",
    "Create a contact card for a software engineer named Alice who is 30 years old.",
    "Return a JSON object with user details.",
    "Provide structured data about a person from Seattle.",
    "Generate a profile for a machine learning engineer.",
]

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "schemas"

ALL_CONFIGS = ["C1", "C2", "C3", "C4"]


# ---------------------------------------------------------------------------
# generate command
# ---------------------------------------------------------------------------
def cmd_generate(args: argparse.Namespace) -> int:
    engine = VeloSpec(
        target_model=args.target,
        draft_model=args.draft,
        config=args.config,
        K=args.K,
    )

    schema = None
    if args.schema:
        schema = args.schema
    elif args.config != "C1":
        print(f"Error: {args.config} requires --schema", file=sys.stderr)
        return 1

    prompt = args.prompt or input("Prompt: ")

    result = engine.generate(prompt, schema=schema, max_tokens=args.max_tokens)

    print(f"\n--- Result ({args.config}) ---")
    print(f"Text:         {result.text}")
    print(f"Acceptance:   {result.acceptance_rate:.2%}")
    print(f"Tokens:       {len(result.token_ids)}")
    print(f"Time:         {result.time_sec:.3f}s")
    print(f"Throughput:   {result.tokens_per_sec:.1f} tok/s")
    if result.k_trace:
        avg_k = sum(k for _, k in result.k_trace) / len(result.k_trace)
        avg_d = sum(d for d, _ in result.k_trace) / len(result.k_trace)
        print(f"Avg K:        {avg_k:.1f}  (avg density={avg_d:.4f})")
    return 0


# ---------------------------------------------------------------------------
# bench command
# ---------------------------------------------------------------------------
def cmd_bench(args: argparse.Namespace) -> int:
    configs = args.configs.split(",")
    schema_names = args.schemas.split(",")

    print("=" * 60)
    print(f"VeloSpec Benchmark: {' + '.join(configs)}")
    print("=" * 60)

    # Load schema files
    schemas_raw: dict[str, str] = {}
    for name in schema_names:
        schema_path = SCHEMA_DIR / f"schema_{name}.json"
        if not schema_path.exists():
            print(f"Error: schema file not found: {schema_path}", file=sys.stderr)
            return 1
        schemas_raw[name] = schema_path.read_text()

    # Prepare output
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "benchmark.csv"

    rows: list[dict] = []
    total_runs = len(configs) * len(schema_names) * len(DEFAULT_PROMPTS)
    run_idx = 0

    for config in configs:
        engine = VeloSpec(
            target_model=args.target,
            draft_model=args.draft,
            config=config,
            K=args.K,
        )

        for schema_name in schema_names:
            schema_str = schemas_raw[schema_name] if config != "C1" else None

            for prompt_idx, prompt in enumerate(DEFAULT_PROMPTS):
                run_idx += 1
                print(f"\n[{run_idx}/{total_runs}] config={config}  schema={schema_name}  prompt={prompt_idx}")

                result = engine.generate(
                    prompt,
                    schema=schema_str,
                    max_tokens=args.max_tokens,
                )

                row = {
                    "config": config,
                    "schema": schema_name,
                    "prompt_id": str(prompt_idx),
                    "acceptance_rate": f"{result.acceptance_rate:.4f}",
                    "total_accepted": result.accepted_count,
                    "total_drafted": result.drafted_count,
                    "time_sec": f"{result.time_sec:.3f}",
                    "tokens_per_sec": f"{result.tokens_per_sec:.2f}",
                    "output_length": len(result.token_ids),
                    "rounds": result.rounds,
                    "avg_density": "",
                    "avg_k": "",
                }

                if result.k_trace:
                    avg_d = sum(d for d, _ in result.k_trace) / len(result.k_trace)
                    avg_k = sum(k for _, k in result.k_trace) / len(result.k_trace)
                    row["avg_density"] = f"{avg_d:.6f}"
                    row["avg_k"] = f"{avg_k:.1f}"

                rows.append(row)

                extra = ""
                if row["avg_k"]:
                    extra = f"  avg_k={row['avg_k']}  density={row['avg_density']}"
                print(f"  accept_rate={row['acceptance_rate']}  "
                      f"time={row['time_sec']}s  output_len={row['output_length']}{extra}")

    # Write CSV
    csv_fields = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[OK] Results saved to {csv_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    from collections import defaultdict
    agg: dict[tuple, list] = defaultdict(list)
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

    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        prog="velospec",
        description="VeloSpec — grammar-aware adaptive speculative decoding",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # generate
    gen_parser = subparsers.add_parser("generate", help="Generate structured output")
    gen_parser.add_argument("prompt", nargs="?", help="Prompt text")
    gen_parser.add_argument("--target", required=True, help="Target model HuggingFace ID")
    gen_parser.add_argument("--draft", required=True, help="Draft model HuggingFace ID")
    gen_parser.add_argument("--config", default="C4", choices=ALL_CONFIGS)
    gen_parser.add_argument("--K", type=int, default=5, help="Speculation width (K_MAX for C4)")
    gen_parser.add_argument("--schema", help="JSON schema file path or raw JSON string")
    gen_parser.add_argument("--max-tokens", type=int, default=256)

    # bench
    bench_parser = subparsers.add_parser("bench", help="Run benchmark")
    bench_parser.add_argument("--target", required=True, help="Target model HuggingFace ID")
    bench_parser.add_argument("--draft", required=True, help="Draft model HuggingFace ID")
    bench_parser.add_argument("--configs", default="C1,C2,C3,C4", help="Comma-separated configs")
    bench_parser.add_argument("--schemas", default="simple,tool_call,nested", help="Comma-separated schemas")
    bench_parser.add_argument("--K", type=int, default=5)
    bench_parser.add_argument("--max-tokens", type=int, default=256)

    args = parser.parse_args()

    if args.command == "generate":
        return cmd_generate(args)
    elif args.command == "bench":
        return cmd_bench(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
