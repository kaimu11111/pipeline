import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# CUDA source (kernel + C++ wrapper, WITHOUT PYBIND11_MODULE)
# -----------------------------------------------------------------------------
source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

/*
 *  High-throughput fused kernel for Turing:
 *
 *      Y = ReLU( (X · W^T + b) - subtract_value ) * multiply_value
 *
 *  • FP32 math with float4 (128-bit) vectorised global-memory transactions.
 *  • 32×32 thread-block (1024 threads) maximises SM occupancy on RTX 6000.
 *  • Each thread computes one output element (row, col).
 *  • K-dimension loop unrolled four times to hide latency.
 *
 *  Assumes row-major inputs:
 *      X : [batch_size,  in_features]
 *      W : [out_features, in_features]
 *      b : [out_features]
 *      Y : [batch_size, out_features]
 */

constexpr int TILE_M = 32;   // rows per block
constexpr int TILE_N = 32;   // cols per block

template<int UNROLL_K>
__global__ void fused_linear_scalar_relu_vec4_kernel(
        const float *__restrict__ X,
        const float *__restrict__ W,
        const float *__restrict__ b,
        const float  subtract_value,
        const float  multiply_value,
        const int    batch_size,
        const int    in_features,
        const int    out_features,
        float *__restrict__ Y)
{
    const int row = blockIdx.y * TILE_M + threadIdx.y; // batch index
    const int col = blockIdx.x * TILE_N + threadIdx.x; // out index

    if (row >= batch_size || col >= out_features) return;

    /* pointers */
    const float *x_ptr = X + row * in_features;
    const float *w_ptr = W + col * in_features;

    float acc = 0.f;

    const int v4_cnt   = in_features >> 2; // /4
    const int tail_idx = v4_cnt << 2;      // *4

#pragma unroll
    for (int k4 = 0; k4 < v4_cnt; k4 += UNROLL_K) {
        #pragma unroll
        for (int uk = 0; uk < UNROLL_K && (k4 + uk) < v4_cnt; ++uk) {
            const float4 x4 = *reinterpret_cast<const float4 *>(x_ptr + ((k4 + uk) << 2));
            const float4 w4 = *reinterpret_cast<const float4 *>(w_ptr + ((k4 + uk) << 2));

            acc += x4.x * w4.x;
            acc += x4.y * w4.y;
            acc += x4.z * w4.z;
            acc += x4.w * w4.w;
        }
    }

    for (int k = tail_idx; k < in_features; ++k)
        acc += x_ptr[k] * w_ptr[k];

    acc = (acc + b[col] - subtract_value) * multiply_value;
    acc = fmaxf(acc, 0.f); // ReLU

    Y[row * out_features + col] = acc;
}

/* -------------------- C++ interface ---------------------*/
torch::Tensor fused_linear_scalar_relu_cuda(
        torch::Tensor X,
        torch::Tensor W,
        torch::Tensor b,
        double subtract_value,
        double multiply_value) {

    TORCH_CHECK(X.is_cuda() && W.is_cuda() && b.is_cuda(),
                "All tensors must be CUDA");
    TORCH_CHECK(X.dtype() == torch::kFloat32 &&
                W.dtype() == torch::kFloat32 &&
                b.dtype() == torch::kFloat32,
                "All tensors must be float32");
    TORCH_CHECK(X.is_contiguous() && W.is_contiguous() && b.is_contiguous(),
                "All tensors must be contiguous");

    const int batch_size   = X.size(0);
    const int in_features  = X.size(1);
    const int out_features = W.size(0);

    TORCH_CHECK(W.size(1) == in_features, "Weight shape mismatch");
    TORCH_CHECK(b.size(0) == out_features, "Bias shape mismatch");

    auto Y = torch::empty({batch_size, out_features}, X.options());

    dim3 block(TILE_N, TILE_M, 1);
    dim3 grid((out_features + TILE_N - 1) / TILE_N,
              (batch_size  + TILE_M - 1) / TILE_M,
              1);

    constexpr int UNROLL_K = 4;
    fused_linear_scalar_relu_vec4_kernel<UNROLL_K><<<grid, block, 0>>>(
        X.data_ptr<float>(),
        W.data_ptr<float>(),
        b.data_ptr<float>(),
        static_cast<float>(subtract_value),
        static_cast<float>(multiply_value),
        batch_size,
        in_features,
        out_features,
        Y.data_ptr<float>());

    return Y;
}
"""

# -----------------------------------------------------------------------------
# C++ prototypes
# -----------------------------------------------------------------------------
cpp_src = """
torch::Tensor fused_linear_scalar_relu_cuda(
    torch::Tensor X,
    torch::Tensor W,
    torch::Tensor b,
    double subtract_value,
    double multiply_value);
"""

# -----------------------------------------------------------------------------
# Build / load extension
# -----------------------------------------------------------------------------
fused_extension = load_inline(
    name="fused_linear_scalar_relu",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["fused_linear_scalar_relu_cuda"],
    verbose=False,
)

# -----------------------------------------------------------------------------
# PyTorch module
# -----------------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Fused implementation:
        Y = ReLU( (X @ W.T + b) - subtract_value ) * multiply_value
    """
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.empty(out_features))

        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        bound = 1 / (in_features ** 0.5)
        nn.init.uniform_(self.bias, -bound, bound)

        self.subtract_value = float(subtract_value)
        self.multiply_value = float(multiply_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_extension.fused_linear_scalar_relu_cuda(
            x.contiguous(),
            self.weight.contiguous(),
            self.bias.contiguous(),
            self.subtract_value,
            self.multiply_value,
        )
