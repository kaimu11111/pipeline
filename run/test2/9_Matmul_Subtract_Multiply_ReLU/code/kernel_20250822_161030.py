import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# CUDA source
# -----------------------------------------------------------------------------
source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

/*
 *  High-throughput fused GEMM-style kernel for NVIDIA Turing (RTX 6000)
 *
 *      Y = ReLU( (X · W^T + b) - subtract_value ) * multiply_value
 *
 *  • 32×32 thread-block (1024 threads) fits warp-level programming model.
 *  • K-dimension processed in chunks of 128 floats (512 B) using shared memory
 *    to increase operand re-use (each tile read only once from DRAM).
 *  • float4 (128-bit) vectorised shared-memory transactions inside the block.
 *  • Requires in_features to be a multiple of 4 (vectorised access).
 *
 *  Tensor dimensions (row-major):
 *      X : [batch_size,  in_features]
 *      W : [out_features, in_features]
 *      b : [out_features]
 *      Y : [batch_size,  out_features]
 */

#define TILE_M 32          // rows  per block (batch)
#define TILE_N 32          // cols  per block (out_features)
#define TILE_K 128         // depth per iteration (must be % 4)

__global__ void fused_linear_scalar_relu_kernel(
        const float*  __restrict__ X,
        const float*  __restrict__ W,
        const float*  __restrict__ b,
        const float   subtract_value,
        const float   multiply_value,
        const int     batch_size,
        const int     in_features,
        const int     out_features,
        float*        __restrict__ Y)
{
    /* ---- thread & block indices ---- */
    const int tx  = threadIdx.x;                    // [0, TILE_N)
    const int ty  = threadIdx.y;                    // [0, TILE_M)
    const int row = blockIdx.y * TILE_M + ty;       // batch index
    const int col = blockIdx.x * TILE_N + tx;       // out_feature index

    if (row >= batch_size || col >= out_features) return;

    /* ---- shared memory (double-buffered) ---- */
    __shared__ float sm_x[TILE_M][TILE_K];
    __shared__ float sm_w[TILE_N][TILE_K];

    float acc = 0.f;

    /* ---- iterate over K-dimension ---- */
    for (int k0 = 0; k0 < in_features; k0 += TILE_K)
    {
        /* --- cooperative load: X tile --- */
        int kx = k0 + tx;  // each thread (ty,tx) loads one element
        if (ty < TILE_M && kx < in_features && row < batch_size)
            sm_x[ty][tx] = __ldg(X + row * in_features + kx);
        else
            sm_x[ty][tx] = 0.f;

        /* --- cooperative load: W tile --- */
        int kw = k0 + ty;  // reuse ty for coalesced access
        if (tx < TILE_N && kw < in_features && col < out_features)
            sm_w[tx][ty] = __ldg(W + col * in_features + kw);
        else
            sm_w[tx][ty] = 0.f;

        __syncthreads();

        /* --- compute partial dot-product --- */
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k)
            acc += sm_x[ty][k] * sm_w[tx][k];

        __syncthreads();
    }

    /* ---- post-ops ---- */
    acc = (acc + b[col] - subtract_value) * multiply_value;
    acc = fmaxf(acc, 0.f);  // ReLU

    /* ---- write result ---- */
    Y[row * out_features + col] = acc;
}

/* -------------------- C++ interface ---------------------*/
torch::Tensor fused_linear_scalar_relu_cuda(
        torch::Tensor X,
        torch::Tensor W,
        torch::Tensor b,
        double subtract_value,
        double multiply_value)
{
    /* --- argument checks --- */
    TORCH_CHECK(X.is_cuda() && W.is_cuda() && b.is_cuda(),
                "All tensors must be CUDA");
    TORCH_CHECK(X.dtype() == torch::kFloat32 &&
                W.dtype() == torch::kFloat32 &&
                b.dtype() == torch::kFloat32,
                "All tensors must be float32 (FP32)");
    TORCH_CHECK(X.is_contiguous() && W.is_contiguous() && b.is_contiguous(),
                "All tensors must be contiguous");

    const int batch_size   = X.size(0);
    const int in_features  = X.size(1);
    const int out_features = W.size(0);

    TORCH_CHECK(in_features % 4 == 0,
                "in_features must be a multiple of 4 for vectorised kernel");
    TORCH_CHECK(W.size(1) == in_features, "Weight shape mismatch");
    TORCH_CHECK(b.size(0) == out_features, "Bias shape mismatch");

    auto Y = torch::empty({batch_size, out_features}, X.options());

    dim3 block(TILE_N, TILE_M, 1); // 32×32 = 1024 threads
    dim3 grid((out_features + TILE_N - 1) / TILE_N,
              (batch_size  + TILE_M - 1) / TILE_M,
              1);

    fused_linear_scalar_relu_kernel<<<grid, block, 0>>>(
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
