// grammar_masked_argmax.cu — Kernel 2: Fused grammar mask + argmax (verify path)
//
// For the verify phase of speculative decoding, we need to:
// 1. Apply grammar bitmask to target logits (set invalid → -inf)
// 2. Find argmax over the masked logits
//
// This kernel fuses both operations into a single pass over the logits array.
// For vocab_size=248320, logits = 248320 × 4B = ~970KB. One pass instead of two.
//
// HPC talking points:
// - One block per position (K+1 positions), threads cooperatively scan vocab
// - Shared memory reduction for parallel argmax within each block
// - Grid-stride loop for large vocab sizes
// - Memory coalescing: consecutive threads access consecutive logits elements
//
// Alternative approach (xgrammar + torch argmax): 2 passes over logits
//   Pass 1: apply_token_bitmask_inplace → read + write 970KB
//   Pass 2: torch.argmax → read 970KB
//   Total: 2.9MB memory traffic
// Our fused kernel: 1 pass, 970KB read + 4B write per position

#include <cuda_runtime.h>
#include <torch/extension.h>

// ============================================================================
// Kernel: grammar_masked_argmax — one block per position
// ============================================================================
//
// Grid: (K+1) blocks, 256 threads per block
// Each block processes one position's logits [vocab_size] with its bitmask
//
// Algorithm:
//   Phase 1: Each thread scans its chunk of vocab, tracks local (max_val, max_idx)
//   Phase 2: Warp-level reduction (shuffle) to find per-warp winner
//   Phase 3: Block-level reduction (shared memory) for final argmax
//
__global__ void grammar_masked_argmax_kernel(
    const float* __restrict__ logits,       // [K+1, vocab_size]
    const int32_t* __restrict__ bitmask,    // [K+1, num_words] where num_words = ceil(V/32)
    int* __restrict__ argmax_indices,       // [K+1] output
    int vocab_size,
    int num_words                           // ceil(vocab_size / 32)
) {
    int pos = blockIdx.x;       // which position (0 to K)
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Phase 1: Each thread scans its chunk, finds local max among valid tokens
    float local_max = -INFINITY;
    int local_max_idx = -1;

    const float* pos_logits = logits + pos * vocab_size;
    const int32_t* pos_mask = bitmask + pos * num_words;

    for (int i = tid; i < vocab_size; i += block_size) {
        // Check if token i is valid in the bitmask
        int word_idx = i / 32;
        int bit_idx = i % 32;
        bool valid = (pos_mask[word_idx] >> bit_idx) & 1;

        if (valid) {
            float val = pos_logits[i];
            if (val > local_max) {
                local_max = val;
                local_max_idx = i;
            }
        }
    }

    // Phase 2: Warp-level argmax reduction via shuffle
    // Reduce 32 threads → 1 per warp
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_max = __shfl_down_sync(0xFFFFFFFF, local_max, offset);
        int other_idx = __shfl_down_sync(0xFFFFFFFF, local_max_idx, offset);
        if (other_max > local_max) {
            local_max = other_max;
            local_max_idx = other_idx;
        }
    }

    // Phase 3: Block-level reduction via shared memory
    // Number of warps = block_size / 32
    int num_warps = block_size / 32;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // Shared memory: each warp writes its winner
    __shared__ float warp_max[8];   // max 8 warps for 256 threads
    __shared__ int warp_idx[8];

    if (lane_id == 0) {
        warp_max[warp_id] = local_max;
        warp_idx[warp_id] = local_max_idx;
    }
    __syncthreads();

    // Final reduction: thread 0 finds global max across all warps
    if (tid == 0) {
        float global_max = -INFINITY;
        int global_idx = -1;
        for (int w = 0; w < num_warps; w++) {
            if (warp_max[w] > global_max) {
                global_max = warp_max[w];
                global_idx = warp_idx[w];
            }
        }
        argmax_indices[pos] = global_idx;
    }
}

// ============================================================================
// Launcher (called from Python via pybind11)
// ============================================================================
void grammar_masked_argmax_launcher(
    const torch::Tensor& logits,       // [K+1, vocab_size] float32, on GPU
    const torch::Tensor& bitmask,      // [K+1, num_words] int32, on GPU
    torch::Tensor& argmax_indices,     // [K+1] int32, on GPU
    int vocab_size,
    int num_words
) {
    TORCH_CHECK(logits.is_cuda(), "logits must be on CUDA device");
    TORCH_CHECK(bitmask.is_cuda(), "bitmask must be on CUDA device");

    int K_plus_1 = logits.size(0);
    int threads = 256;
    int blocks = K_plus_1;

    grammar_masked_argmax_kernel<<<blocks, threads>>>(
        logits.data_ptr<float>(),
        bitmask.data_ptr<int32_t>(),
        argmax_indices.data_ptr<int>(),
        vocab_size,
        num_words
    );
}
