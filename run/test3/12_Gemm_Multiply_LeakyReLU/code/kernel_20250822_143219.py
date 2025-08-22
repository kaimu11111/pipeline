import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# --------------------------------------------------------------------------------
# CUDA source code: fused GEMM + bias add + scalar multiply + LeakyReLU
# --------------------------------------------------------------------------------
source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

////////////////////////////////////////////////////////////////////////////////
// Tiled GEMM kernel with bias, scalar multiply, and LeakyReLU fused in.
////////////////////////////////////////////////////////////////////////////////
const int TILE = 16;

__global__ void fused_gemm_bias_mul_leakyrelu_kernel(
        const float* __restrict__ A,   // [M, K]
        const float* __restrict__ B,   // [N, K]  (row-major, not transposed)
        const float* __restrict__ bias,// [N]
        float* __restrict__ C,         // [M, N]
        int M, int N, int K,
        float multiplier,
        float negative_slope)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    // Row and column this thread block will compute
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float value = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t)
    {
        // Load A tile (row-major)
        int tiled_row = row;
        int tiled_col = t * TILE + threadIdx.x;
        if (tiled_row < M && tiled_col < K)
            As[threadIdx.y][threadIdx.x] = A[tiled_row * K + tiled_col];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Load B tile
        int tiled_row_B = col;
        int tiled_col_B = t * TILE + threadIdx.y;
        if (tiled_row_B < N && tiled_col_B < K)
            Bs[threadIdx.y][threadIdx.x] = B[tiled_row_B * K + tiled_col_B];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial product
        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            value += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    // Write the result with bias, multiplier, and LeakyReLU
    if (row < M && col < N)
    {
        float out_val = (value + bias[col]) * multiplier;
        out_val = (out_val >= 0.0f) ? out_val : out_val * negative_slope;
        C[row * N + col] = out_val;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Host wrapper
////////////////////////////////////////////////////////////////////////////////
torch::Tensor fused_gemm_bias_mul_leakyrelu_cuda(
        torch::Tensor A,      // [M, K], float32, contiguous CUDA
        torch::Tensor B,      // [N, K], float32, contiguous CUDA
        torch::Tensor bias,   // [N],    float32, contiguous CUDA
        float multiplier,
        float negative_slope)
{
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && bias.is_cuda(), "All tensors must be on CUDA");
    TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32 && bias.dtype() == torch::kFloat32,
                "All tensors must be float32");

    int64_t M = A.size(0);
    int64_t K = A.size(1);
    TORCH_CHECK(B.size(1) == K, "Incompatible shapes for GEMM (K dim)");
    int64_t N = B.size(0);
    TORCH_CHECK(bias.numel() == N, "Bias size must match N dimension");

    auto C = torch::empty({M, N}, A.options());

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    fused_gemm_bias_mul_leakyrelu_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        bias.data_ptr<float>(),
        C.data_ptr<float>(),
        (int)M, (int)N, (int)K,
        multiplier,
        negative_slope);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_gemm_bias_mul_leakyrelu_cuda", &fused_gemm_bias_mul_leakyrelu_cuda,
          "Fused GEMM + bias + mul + LeakyReLU (CUDA)");
}
"""

# --------------------------------------------------------------------------------
# C++ prototypes exposed to Python
# --------------------------------------------------------------------------------
cpp_src = """
torch::Tensor fused_gemm_bias_mul_leakyrelu_cuda(
        torch::Tensor A,
        torch::Tensor B,
        torch::Tensor bias,
        float multiplier,
        float negative_slope);
"""

# --------------------------------------------------------------------------------
# Compile and load the extension
# --------------------------------------------------------------------------------
fused_ext = load_inline(
    name="fused_gemm_bias_mul_leakyrelu",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["fused_gemm_bias_mul_leakyrelu_cuda"],
    verbose=False,
)

# --------------------------------------------------------------------------------
# Optimised PyTorch module using the custom CUDA kernel
# --------------------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Re-implementation of the original Model with all PyTorch ops replaced
    by hand-written CUDA kernels.
    """
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super().__init__()
        # Parameters equivalent to nn.Linear
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.empty(out_features))
        self.multiplier = float(multiplier)
        self.negative_slope = float(negative_slope)
        # Kaiming uniform initialisation (similar to nn.Linear default)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = in_features
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Ensure contiguous tensors on CUDA
        if not x.is_cuda:
            x = x.cuda()
        x = x.contiguous()
        weight = self.weight.contiguous().cuda()
        bias   = self.bias.contiguous().cuda()
        # Call fused CUDA kernel
        return fused_ext.fused_gemm_bias_mul_leakyrelu_cuda(
            x, weight, bias, self.multiplier, self.negative_slope)
import math
