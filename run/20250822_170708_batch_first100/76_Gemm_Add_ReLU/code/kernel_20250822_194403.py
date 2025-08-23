import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Handwritten fused GEMM+Bias+ReLU CUDA kernel
source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void gemm_bias_relu_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ C,
    int M, int K, int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float val = 0.0f;
        // Naive GEMM: A: [M, K], B: [N, K], C: [M, N]
        // B is shaped [out_features, in_features],
        // but we interpret it as [N, K] because out_features=N, in_features=K.
        for (int i = 0; i < K; i++) {
            val += A[row * K + i] * B[col * K + i];
        }
        // Add bias
        val += bias[col];
        // Apply ReLU
        val = fmaxf(val, 0.0f);
        C[row * N + col] = val;
    }
}

torch::Tensor gemm_bias_relu_cuda(
    torch::Tensor A,   // [M, K]
    torch::Tensor B,   // [N, K]
    torch::Tensor bias // [N]
) {
    auto M = A.size(0);
    auto K = A.size(1);
    auto N = B.size(0);

    // Make inputs contiguous
    auto A_contig = A.contiguous();
    auto B_contig = B.contiguous();
    auto bias_contig = bias.contiguous();

    // Prepare output
    auto options = torch::TensorOptions().dtype(A.dtype()).device(A.device());
    auto C = torch::zeros({M, N}, options);

    // Launch config
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    gemm_bias_relu_kernel<<<grid, block>>>(
        A_contig.data_ptr<float>(),
        B_contig.data_ptr<float>(),
        bias_contig.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, N
    );

    return C;
}
"""

cpp_src = r"""
torch::Tensor gemm_bias_relu_cuda(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor bias
);
"""

# Build the inline extension
gemmBiasRelu = load_inline(
    name="gemm_bias_relu",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["gemm_bias_relu_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    """
    Optimized model that fuses GEMM, bias addition, and ReLU into a single custom CUDA kernel.
    """
    def __init__(self, in_features, out_features, bias_shape):
        super(ModelNew, self).__init__()
        # Matches original parameter naming/shape
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(*bias_shape))

    def forward(self, x):
        # Invoke the fused custom kernel
        return gemmBiasRelu.gemm_bias_relu_cuda(x, self.weight, self.bias)
