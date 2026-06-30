// bindings.cpp — Pybind11 / torch extension bindings for GrammarSD CUDA kernels
//
// This file declares the Python-facing API for our 3 CUDA kernels.
// Built via setup.py with torch.utils.cpp_extension.CUDAExtension.
//
// Usage from Python:
//   import grammar_sd_kernels
//   grammar_sd_kernels.popcount_density(bitmask, total_count, num_words)
//   grammar_sd_kernels.grammar_masked_argmax(logits, bitmask, argmax_out, vocab_size, num_words)
//   grammar_sd_kernels.fused_sample(logits, bitmask, sampled_out, temp, vocab_size, num_words, seed)

#include <pybind11/pybind11.h>
#include <torch/extension.h>

// Declare launchers (defined in .cu files)
void popcount_density_launcher(
    const torch::Tensor& bitmask,
    torch::Tensor& total_count,
    int num_words
);

void grammar_masked_argmax_launcher(
    const torch::Tensor& logits,
    const torch::Tensor& bitmask,
    torch::Tensor& argmax_indices,
    int vocab_size,
    int num_words
);

void fused_sample_launcher(
    const torch::Tensor& logits,
    const torch::Tensor& bitmask,
    torch::Tensor& sampled_token,
    float temperature,
    int vocab_size,
    int num_words,
    unsigned long long seed
);

// ============================================================================
// Python-facing wrappers with input validation
// ============================================================================

void py_popcount_density(
    torch::Tensor bitmask,
    torch::Tensor total_count,
    int64_t num_words
) {
    TORCH_CHECK(bitmask.is_cuda(), "bitmask must be on CUDA");
    TORCH_CHECK(bitmask.dtype() == torch::kInt32, "bitmask must be int32");
    TORCH_CHECK(total_count.is_cuda(), "total_count must be on CUDA");
    popcount_density_launcher(bitmask, total_count, (int)num_words);
}

void py_grammar_masked_argmax(
    torch::Tensor logits,
    torch::Tensor bitmask,
    torch::Tensor argmax_indices,
    int64_t vocab_size,
    int64_t num_words
) {
    TORCH_CHECK(logits.is_cuda(), "logits must be on CUDA");
    TORCH_CHECK(bitmask.is_cuda(), "bitmask must be on CUDA");
    grammar_masked_argmax_launcher(logits, bitmask, argmax_indices, (int)vocab_size, (int)num_words);
}

void py_fused_sample(
    torch::Tensor logits,
    torch::Tensor bitmask,
    torch::Tensor sampled_token,
    double temperature,
    int64_t vocab_size,
    int64_t num_words,
    int64_t seed
) {
    TORCH_CHECK(logits.is_cuda(), "logits must be on CUDA");
    TORCH_CHECK(bitmask.is_cuda(), "bitmask must be on CUDA");
    fused_sample_launcher(
        logits, bitmask, sampled_token,
        (float)temperature, (int)vocab_size, (int)num_words,
        (unsigned long long)seed
    );
}

// ============================================================================
// Module definition
// ============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "GrammarSD CUDA kernels: popcount density, grammar-masked argmax, fused sampling";

    m.def("popcount_density", &py_popcount_density,
          "Count set bits in grammar bitmask (Kernel 1)",
          pybind11::arg("bitmask"), pybind11::arg("total_count"), pybind11::arg("num_words"));

    m.def("grammar_masked_argmax", &py_grammar_masked_argmax,
          "Fused grammar mask + argmax over K+1 positions (Kernel 2)",
          pybind11::arg("logits"), pybind11::arg("bitmask"),
          pybind11::arg("argmax_indices"), pybind11::arg("vocab_size"), pybind11::arg("num_words"));

    m.def("fused_sample", &py_fused_sample,
          "Fused grammar mask + softmax + sample for single position (Kernel 3)",
          pybind11::arg("logits"), pybind11::arg("bitmask"),
          pybind11::arg("sampled_token"), pybind11::arg("temperature"),
          pybind11::arg("vocab_size"), pybind11::arg("num_words"), pybind11::arg("seed"));
}
