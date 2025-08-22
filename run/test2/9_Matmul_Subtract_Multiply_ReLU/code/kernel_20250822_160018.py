import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# CUDA source (kernel + C++ wrapper)
# -----------------------------------------------------------------------------
source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

/*
 *  Fused kernel:
 *  Y = ReLU( (X * W^T + b) - subtract_value ) * multiply_value
 *
 *  Vectorised over K dimension with float4 loads for better memory throughput.
 *  Works for any in_features; tail-processing handles non-multiples of 4.
 */

constexpr int TILE_M = 32;   // rows  per block  (batch)
constexpr int TILE_N = 32;   // cols  per block  (out_features)

/* kernel */
__global__ void fused_linear_scalar_relu_kernel_v4(
        const float* __restrict__ X,
        const float* __restrict__ W,
        const float* __restrict__ b,
        const float  subtract_value,
        const float  multiply_value,
        const int    batch_size,
        const int    in_features,
        const int    out_features,
        float* __restrict__ Y) {

    const int row = blockIdx.y * TILE_M + threadIdx.y; // batch idx
    const int col = blockIdx.x * TILE_N + threadIdx.x; // out   idx

    if (row >= batch_size || col >= out_features) return;

    float acc = 0.f;

    const int k_full_steps = in_features / 4;        // processed with float4
    const int k_tail_start = k_full_steps * 4;       // remaining (<4)

    const float* x_ptr = X + row * in_features;
    const float* w_ptr = W + col * in_features;

    /* -------- main loop, 4 floats each iteration ----------*/
    #pragma unroll 4
    for (int k4 = 0; k4 < k_full_steps; ++k4) {
        float4 xv4 = *reinterpret_cast<const float4 const*>(x_ptr + k4 * 4);
        float4 wv4 = *reinterpret_cast<const float4 const*>(w_ptr + k4 * 4);

        acc += xv4.x * wv4.x;
        acc += xv4.y * wv4.y;
        acc += xv4.z * wv4.z;
        acc += xv4.w * wv4.w;
    }

    /* -------------------- tail ----------------------------*/
    for (int k = k_tail_start; k < in_features; ++k)
        acc += x_ptr[k] * w_ptr[k];

    /* bias + scalar ops + ReLU */
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

    dim3 block(TILE_N, TILE_M, 1);           // 32×32 = 1024 threads (Turing max)
    dim3 grid((out_features + TILE_N - 1) / TILE_N,
              (batch_size  + TILE_M - 1) / TILE_M,
              1);

    fused_linear_scalar_relu_kernel_v4<<<grid, block>>>(
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
# C++ prototypes for PyBind11
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
    Fused implementation of:
        Y = ReLU( (X @ W.T + b) - subtract_value ) * multiply_value
    """
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.empty(out_features))

        # Initialize identical to nn.Linear
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
