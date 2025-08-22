import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# CUDA source (kernel + C++ interface, NO pybind11 module definition)
# -----------------------------------------------------------------------------
source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

/*
 *  Tiled fused kernel:
 *  Y = ReLU( (X * W^T + b) - subtract_value ) * multiply_value
 *
 *  A block computes a TILE_M × TILE_N output tile.
 *  K dimension is iterated in chunks of TILE_K, using shared memory to
 *  reuse loaded operands and improve memory-bandwidth utilisation.
 *
 *  Tuned for Turing (Quadro RTX 6000):
 *      TILE_M = 16, TILE_N = 16, TILE_K = 128
 *  => 256 threads / block, 12 KB shared memory.
 */

#define TILE_M 16
#define TILE_N 16
#define TILE_K 128

__global__ void fused_linear_scalar_relu_tiled_kernel(
        const float* __restrict__ X,
        const float* __restrict__ W,
        const float* __restrict__ b,
        const float  subtract_value,
        const float  multiply_value,
        const int    batch_size,
        const int    in_features,
        const int    out_features,
        float* __restrict__ Y) {

    /* --- thread/block indices --- */
    const int tx = threadIdx.x;           // [0 .. TILE_N)
    const int ty = threadIdx.y;           // [0 .. TILE_M)
    const int row = blockIdx.y * TILE_M + ty;   // batch idx
    const int col = blockIdx.x * TILE_N + tx;   // out_feature idx

    /* shared memory allocation */
    __shared__ float sm_x[TILE_M][TILE_K];
    __shared__ float sm_w[TILE_N][TILE_K];

    float acc = 0.f;

    /* iterate over K dimension */
    for (int k0 = 0; k0 < in_features; k0 += TILE_K) {

        /* load X tile: each thread loads one element if within bounds */
        int k_idx = k0 + tx;    /* reuse tx for vectorised coalesced loads */
        if (row < batch_size && k_idx < in_features && ty < TILE_M)
            sm_x[ty][tx] = X[row * in_features + k_idx];
        else
            sm_x[ty][tx] = 0.f;

        /* load W tile: each thread loads one element */
        int k_w = k0 + ty;      /* reuse ty */
        if (col < out_features && k_w < in_features && tx < TILE_N)
            sm_w[tx][ty] = W[col * in_features + k_w];
        else
            sm_w[tx][ty] = 0.f;

        __syncthreads();

        /* compute partial dot product */
        #pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk)
            acc += sm_x[ty][kk] * sm_w[tx][kk];

        __syncthreads();
    }

    /* write result with post-ops */
    if (row < batch_size && col < out_features) {
        float out = (acc + b[col] - subtract_value) * multiply_value;
        out = fmaxf(out, 0.f);
        Y[row * out_features + col] = out;
    }
}

/* -------------------- C++ interface ---------------------*/
torch::Tensor fused_linear_scalar_relu_cuda(
        torch::Tensor X,
        torch::Tensor W,
        torch::Tensor b,
        double subtract_value,
        double multiply_value) {

    /* --- sanity checks --- */
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

    dim3 block(TILE_N, TILE_M, 1);   // 16×16=256 threads
    dim3 grid((out_features + TILE_N - 1) / TILE_N,
              (batch_size  + TILE_M - 1) / TILE_M,
              1);

    fused_linear_scalar_relu_tiled_kernel<<<grid, block>>>(
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
# PyTorch module wrapper
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

        # Init identical to nn.Linear
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
