import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# --------------------------------------------------------------------------------
# CUDA source: fused GEMM + bias add + LeakyReLU + scalar multiply
# --------------------------------------------------------------------------------
source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

////////////////////////////////////////////////////////////////////////////////
// Fused GEMM kernel
////////////////////////////////////////////////////////////////////////////////
const int TILE = 16;

__global__ void fused_gemm_bias_act_mul_kernel(
        const float* __restrict__ A,   // [M, K]
        const float* __restrict__ B,   // [N, K]  (row–major)
        const float* __restrict__ bias,// [N]
        float* __restrict__ C,         // [M, N]
        int M, int N, int K,
        float multiplier,
        float negative_slope)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;   // output row   (m)
    int col = blockIdx.x * TILE + threadIdx.x;   // output col   (n)

    float acc = 0.0f;

    // Iterate over tiles of K dimension
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t)
    {
        // ----- load A tile -----
        int a_col = t * TILE + threadIdx.x;
        if (row < M && a_col < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // ----- load B tile -----
        int b_row = col;                       // because B shape is [N, K]
        int b_col = t * TILE + threadIdx.y;
        if (b_row < N && b_col < K)
            Bs[threadIdx.y][threadIdx.x] = B[b_row * K + b_col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // ----- compute partial -----
        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    // ----- write result -----
    if (row < M && col < N)
    {
        float out = acc + bias[col];                   // add bias
        out = (out >= 0.0f) ? out : out * negative_slope; // LeakyReLU
        out *= multiplier;                            // scalar multiply
        C[row * N + col] = out;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Host wrapper
////////////////////////////////////////////////////////////////////////////////
torch::Tensor fused_gemm_bias_act_mul_cuda(
        torch::Tensor A,
        torch::Tensor B,
        torch::Tensor bias,
        float multiplier,
        float negative_slope)
{
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && bias.is_cuda(), "Tensors must be CUDA");
    TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32
                && bias.dtype() == torch::kFloat32, "Tensors must be float32");

    const int64_t M = A.size(0);
    const int64_t K = A.size(1);
    TORCH_CHECK(B.size(1) == K, "Dimension mismatch: K");
    const int64_t N = B.size(0);
    TORCH_CHECK(bias.numel() == N, "Bias length mismatch");

    auto C = torch::empty({M, N}, A.options());

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    fused_gemm_bias_act_mul_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        bias.data_ptr<float>(),
        C.data_ptr<float>(),
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
        multiplier,
        negative_slope);

    return C;
}
"""

# --------------------------------------------------------------------------------
# C++ prototypes
# --------------------------------------------------------------------------------
cpp_src = """
torch::Tensor fused_gemm_bias_act_mul_cuda(
        torch::Tensor A,
        torch::Tensor B,
        torch::Tensor bias,
        float multiplier,
        float negative_slope);
"""

# --------------------------------------------------------------------------------
# Build & load
# --------------------------------------------------------------------------------
fused_ext = load_inline(
    name="fused_gemm_bias_act_mul",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["fused_gemm_bias_act_mul_cuda"],
    verbose=False,
)

# --------------------------------------------------------------------------------
# PyTorch module
# --------------------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    nn.Linear (x @ W.T + b) followed by LeakyReLU, then scalar multiply.
    Entire sequence fused inside a custom CUDA kernel.
    """
    def __init__(self, in_features, out_features,
                 multiplier=1.0, negative_slope=0.01):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.empty(out_features))
        self.multiplier      = float(multiplier)
        self.negative_slope  = float(negative_slope)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = in_features
        bound = 1.0 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if not x.is_cuda:
            x = x.cuda()
        C = fused_ext.fused_gemm_bias_act_mul_cuda(
            x.contiguous(),
            self.weight.contiguous().cuda(),
            self.bias.contiguous().cuda(),
            self.multiplier,
            self.negative_slope
        )
        return C
