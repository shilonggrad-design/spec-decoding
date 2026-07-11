"""
VeloSpec — Velocity-Optimized Speculative Decoding for Structured Output.

Grammar-aware adaptive speculative decoding that uses grammar mask density
as a forward-looking signal to dynamically adjust speculation width K.

Quickstart:
    from velospec import VeloSpec

    engine = VeloSpec(
        target_model="Qwen/Qwen3.5-4B",
        draft_model="Qwen/Qwen3.5-0.8B",
        config="C4",
    )
    result = engine.generate(prompt="...", schema={...})
    print(result.text, result.acceptance)
"""

__version__ = "0.2.0"

# Lazy imports — avoids hard dependency on xgrammar/torch for kernel-only usage
def __getattr__(name: str):
    if name in ("VeloSpec", "GenerationResult"):
        from velospec.engine import VeloSpec, GenerationResult
        return {"VeloSpec": VeloSpec, "GenerationResult": GenerationResult}[name]
    raise AttributeError(f"module 'velospec' has no attribute {name!r}")

__all__ = ["VeloSpec", "GenerationResult"]
