#!/usr/bin/env python3
"""
quickstart.py — VeloSpec 10-line demo.

Run in Colab after:
    !pip install -e .
    # or on CPU: just python examples/quickstart.py
"""

from velospec import VeloSpec

# 1. Create engine with any HuggingFace model pair
engine = VeloSpec(
    target_model="Qwen/Qwen3.5-4B",
    draft_model="Qwen/Qwen3.5-0.8B",
    config="C4",  # Grammar-guided draft + adaptive K
)

# 2. Generate structured output with a JSON schema
result = engine.generate(
    prompt="Call a function to search for AI news.",
    schema={
        "type": "object",
        "properties": {
            "function": {"type": "string", "enum": ["search", "get", "post"]},
            "arguments": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        },
        "required": ["function", "arguments"],
    },
    max_tokens=256,
)

# 3. Check results
print(f"Output:       {result.text}")
print(f"Acceptance:   {result.acceptance_rate:.2%}")
print(f"Throughput:   {result.tokens_per_sec:.1f} tok/s")
if result.k_trace:
    avg_k = sum(k for _, k in result.k_trace) / len(result.k_trace)
    print(f"Avg K:        {avg_k:.1f} (adaptive)")
