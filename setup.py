"""
setup.py — Build CUDA extension for GrammarSD kernels.

Usage (Colab):
    !pip install ninja  # faster builds
    !python setup.py build_ext --inplace

This compiles 3 CUDA kernels into a single Python extension module:
    1. popcount_density    — count valid tokens in grammar bitmask
    2. grammar_masked_argmax — fused mask + argmax (verify path)
    3. fused_sample         — fused mask + softmax + sample (draft path)

After building, import as:
    import grammar_sd_kernels
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="grammar_sd_kernels",
    ext_modules=[
        CUDAExtension(
            name="grammar_sd_kernels",
            sources=[
                "src/_kernels/popcount_density.cu",
                "src/_kernels/grammar_masked_argmax.cu",
                "src/_kernels/fused_sample.cu",
                "src/_kernels/bindings.cpp",
            ],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
