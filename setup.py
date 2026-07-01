"""
setup.py — Build CUDA extension for VeloSpec kernels.

Usage (Colab):
    !pip install ninja
    !python setup.py build_ext --inplace

This compiles 3 CUDA kernels into a single Python extension module:
    1. popcount_density    — count valid tokens in grammar bitmask
    2. grammar_masked_argmax — fused mask + argmax (verify path)
    3. fused_sample         — fused mask + softmax + sample (draft path)

After building, import as:
    import velospec_kernels
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="velospec_kernels",
    ext_modules=[
        CUDAExtension(
            name="velospec_kernels",
            sources=[
                "velospec/kernels/popcount_density.cu",
                "velospec/kernels/grammar_masked_argmax.cu",
                "velospec/kernels/fused_sample.cu",
                "velospec/kernels/bindings.cpp",
            ],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
