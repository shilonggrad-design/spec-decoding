// popcount_density.cu — Kernel 1: Grammar mask density computation
//
// Counts set bits in grammar bitmask to compute density = valid_tokens / vocab_size.
// This drives the adaptive K controller (C4): low density → large K, high density → small K.
//
// HPC talking points:
// - __popc is a single-instruction hardware intrinsic for int32 popcount
// - Warp shuffle reduction (__shfl_down_sync) avoids shared memory round-trip
// - bitmask = ceil(248320/32) = 7760 × int32 ≈ 31KB → fits in L2 cache
// - Grid-stride loop handles arbitrary vocab sizes
// - Total latency: < 1μs on any modern GPU (latency-bound, not throughput-bound)
//
// Profiling plan: Nsight Compute
// - Compare vs Python loop (current implementation, ~2ms for 7760 words)
// - Expected speedup: ~1000× (hardware popcount + warp reduction)
// - Memory throughput: bitmask is tiny, bandwidth is not the bottleneck

#include <cuda_runtime.h>
#include <torch/extension.h>

// ============================================================================
// Kernel: popcount density — counts valid tokens in grammar bitmask
// ============================================================================
//
// Grid: 1 block, 256 threads (single warp group for small bitmask)
// Each thread processes ceil(7760/256) ≈ 31 words via grid-stride loop
//
__global__ void popcount_density_kernel(
    const int32_t* __restrict__ bitmask,  // ceil(vocab_size / 32) elements
    int* __restrict__ total_count,         // scalar output (atomic accumulation)
    int num_words                          // bitmask length
) {
    int local_count = 0;

    // Grid-stride loop: each thread accumulates its portion
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < num_words;
         i += gridDim.x * blockDim.x) {
        local_count += __popc(bitmask[i]);  // int32 popcount: 1 instruction
    }

    // Warp-level reduction via shuffle (5 instructions for 32→1)
    // Each lane adds its value to the lane `offset` positions below it
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_count += __shfl_down_sync(0xFFFFFFFF, local_count, offset);
    }

    // Lane 0 of each warp atomic-adds to global counter
    // For our 256-thread / 8-warp config, this is 8 atomic operations total
    if (threadIdx.x % 32 == 0) {
        atomicAdd(total_count, local_count);
    }
}

// ============================================================================
// Launcher (called from Python via pybind11)
// ============================================================================
void popcount_density_launcher(
    const torch::Tensor& bitmask,    // [num_words] int32, on GPU
    torch::Tensor& total_count,      // [1] int32, on GPU, must be zeroed
    int num_words
) {
    TORCH_CHECK(bitmask.is_cuda(), "bitmask must be on CUDA device");
    TORCH_CHECK(bitmask.dtype() == torch::kInt32, "bitmask must be int32");
    TORCH_CHECK(total_count.is_cuda(), "total_count must be on CUDA device");

    // 256 threads = 8 warps, enough for 7760 elements
    // Single block is sufficient — this is a tiny reduction
    int threads = 256;
    int blocks = 1;

    popcount_density_kernel<<<blocks, threads>>>(
        bitmask.data_ptr<int32_t>(),
        total_count.data_ptr<int>(),
        num_words
    );
}

// ============================================================================
// C API for bindings.cpp
// ============================================================================
int cuda_popcount_density(
    const int32_t* bitmask_ptr,
    int* total_count_ptr,
    int num_words,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = 1;

    popcount_density_kernel<<<blocks, threads, 0, stream>>>(
        bitmask_ptr,
        total_count_ptr,
        num_words
    );

    return 0;  // success
}
