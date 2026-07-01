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

from velospec.engine import VeloSpec, GenerationResult

__version__ = "0.2.0"
__all__ = ["VeloSpec", "GenerationResult"]
