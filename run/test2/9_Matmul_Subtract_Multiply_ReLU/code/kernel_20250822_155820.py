import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# CUDA source (kernel + C++ wrapper—without PyBind11 module declaration)
# -----------------------------------------------------------------------------
source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

/*
 *  Fused kernel:
 *  Y = ReLU( (X * W^T + b) - subtract_value ) * multiply_value
 *
 *  X : [batch_size, in_features]          (row-major)
 *  W : [out_features, in_features]        (row-major)
 *  b : [out_features]
 *  Y : [batch_size, out_features]         (row-major)
 */

template<int TILE_K>
__global__ void fused_linear_scalar_relu_kernel(
        const float* __restrict__ X,
        const float* __restrict__ W,
        const float* __restrict__ b,
        const float  subtract_value,
        const float  multiply_value,
        const int    batch_size,
        const int    in_features,
        const int    out_features,
        float* __restrict__ Y) {

    // 2-D thread indices
    const int row = blockIdx.y * blockDim.y + threadIdx.y; // batch dimension
    const int col = blockIdx.x * blockDim.x + threadIdx.x; // out_features dimension

    if (row >= batch_size || col >= out_features) return;

    float acc = 0.0f;

    // Dot-product accumulation
    for (int k_base = 0; k_base < in_features; k_base += TILE_K) {
        const int k_limit = min(TILE_K, in_features - k_base);
        #pragma unroll
        for (int k = 0; k < k_limit; ++k) {
            float x_val = X[row * in_features + (k_base + k)];
            float w_val = W[col * in_features + (k_base + k)];
            acc += x_val * w_val;
        }
    }

    // Bias, scalar ops, and ReLU
    acc = (acc + b[col] - subtract_value) * multiply_value;
    if (acc < 0.0f) acc = 0.0f;

    Y[row * out_features + col] = acc;
}

torch::Tensor fused_linear_scalar_relu_cuda(
        torch::Tensor X,
        torch::Tensor W,
        torch::Tensor b,
        double subtract_value,
        double multiply_value) {

    TORCH_CHECK(X.is_cuda(), "X must be a CUDA tensor");
    TORCH_CHECK(W.is_cuda(), "W must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(X.dtype() == torch::kFloat32, "X must be float32");
    TORCH_CHECK(W.dtype() == torch::kFloat32, "W must be float32");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "b must be float32");
    TORCH_CHECK(X.is_contiguous(), "X must be contiguous");
    TORCH_CHECK(W.is_contiguous(), "W must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "b must be contiguous");

    const int batch_size   = X.size(0);
    const int in_features  = X.size(1);
    const int out_features = W.size(0);
    TORCH_CHECK(W.size(1) == in_features, "Weight shape incompatible");
    TORCH_CHECK(b.size(0) == out_features, "Bias shape incompatible");

    auto Y = torch::empty({batch_size, out_features}, X.options());

    const int TILE_M = 16;   // rows  per block
    const int TILE_N = 16;   // cols  per block
    const int TILE_K = 8;    // depth loop unrolling

    dim3 block(TILE_N, TILE_M, 1);
    dim3 grid((out_features + TILE_N - 1) / TILE_N,
              (batch_size  + TILE_M - 1) / TILE_M,
              1);

    fused_linear_scalar_relu_kernel<TILE_K><<<grid, block>>>(
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
# C++ function prototype for automatic PyBind11 generation
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
# Build / load
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
    Fused implementation of:
        Y = ReLU( (X @ W.T + b) - subtract_value ) * multiply_value
    """
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.empty(out_features))

        # Initialization matching nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        fan_in = in_features
        bound = 1 / (fan_in ** 0.5)
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
