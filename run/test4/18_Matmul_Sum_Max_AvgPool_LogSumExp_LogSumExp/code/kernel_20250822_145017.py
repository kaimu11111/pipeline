import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------------
# CUDA source: kernels + C++/CUDA wrappers
# NOTE: Do NOT create a PYBIND11_MODULE here; load_inline will
#       auto-generate it from the provided prototypes.
# ------------------------------------------------------------------
source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/*
 * Kernel 1
 *   - Computes:
 *       weight_sums[k]   = Σ_j W[j, k]
 *       bias_sum_scalar  = Σ_j bias[j]
 *   W layout: (out_features, in_features) row–major (same as PyTorch)
 */
__global__ void compute_weight_bias_sums_kernel(const float* __restrict__ W,
                                                const float* __restrict__ bias,
                                                float* __restrict__ weight_sums,
                                                float* __restrict__ bias_sum,
                                                int out_features,
                                                int in_features) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < in_features) {
        float w_sum = 0.f;
        for (int row = 0; row < out_features; ++row) {
            w_sum += W[row * in_features + col];
        }
        weight_sums[col] = w_sum;
    }

    // Thread 0 of block 0 performs bias reduction
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float b_sum = 0.f;
        for (int row = 0; row < out_features; ++row) {
            b_sum += bias[row];
        }
        *bias_sum = b_sum;
    }
}

/*
 * Kernel 2
 *   - Computes:
 *       out[i] = Σ_k X[i, k] * weight_sums[k] + bias_sum
 *   X layout: (batch, in_features) row–major
 */
__global__ void fused_forward_kernel(const float* __restrict__ X,
                                     const float* __restrict__ weight_sums,
                                     const float* __restrict__ bias_sum,
                                     float* __restrict__ out,
                                     int batch,
                                     int in_features) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < batch) {
        float acc = 0.f;
        const float* x_ptr = X + row * in_features;
        for (int k = 0; k < in_features; ++k) {
            acc += x_ptr[k] * weight_sums[k];
        }
        acc += *bias_sum;
        out[row] = acc;
    }
}

/*
 * Host wrapper exposed to Python
 */
torch::Tensor model_fwd_cuda(torch::Tensor X,
                             torch::Tensor W,
                             torch::Tensor bias) {
    CHECK_INPUT(X);
    CHECK_INPUT(W);
    CHECK_INPUT(bias);

    const int batch        = X.size(0);
    const int in_features  = X.size(1);
    const int out_features = W.size(0);

    auto options     = X.options();
    auto weight_sums = torch::empty({in_features}, options);
    auto bias_sum    = torch::empty({1}, options);
    auto out         = torch::empty({batch}, options);

    constexpr int THREADS = 256;
    const int blocks_cols = (in_features + THREADS - 1) / THREADS;
    const int blocks_rows = (batch       + THREADS - 1) / THREADS;

    // Kernel 1: column sums & bias sum
    compute_weight_bias_sums_kernel<<<blocks_cols, THREADS>>>(
        W.data_ptr<float>(),
        bias.data_ptr<float>(),
        weight_sums.data_ptr<float>(),
        bias_sum.data_ptr<float>(),
        out_features,
        in_features);

    // Kernel 2: final fused forward
    fused_forward_kernel<<<blocks_rows, THREADS>>>(
        X.data_ptr<float>(),
        weight_sums.data_ptr<float>(),
        bias_sum.data_ptr<float>(),
        out.data_ptr<float>(),
        batch,
        in_features);

    CUDA_CHECK_ERRORS();
    return out;
}
"""

# ------------------------------------------------------------------
# C++ declaration(s) to expose
# ------------------------------------------------------------------
cpp_src = """
torch::Tensor model_fwd_cuda(torch::Tensor X,
                             torch::Tensor W,
                             torch::Tensor bias);
"""

# ------------------------------------------------------------------
# Compile & load kernels
# ------------------------------------------------------------------
model_cuda = load_inline(
    name="model_cuda_kernels",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["model_fwd_cuda"],
    verbose=False,
)

# ------------------------------------------------------------------
# Optimised PyTorch module using the custom CUDA kernels
# ------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            x = x.cuda()
        x = x.contiguous()

        y = model_cuda.model_fwd_cuda(
            x,
            self.linear.weight.contiguous(),
            self.linear.bias.contiguous(),
        )
        return y.view(-1, 1)
