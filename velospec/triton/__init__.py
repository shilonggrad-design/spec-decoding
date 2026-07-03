"""Triton kernels for VeloSpec."""
try:
    from velospec.triton.fused_logit_processor import fused_masked_argmax
    __all__ = ["fused_masked_argmax"]
except ImportError:
    __all__ = []
