import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Inline fused GEMM + multiply + LeakyReLU CUDA kernel
source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_gemm_mul_leakyrelu_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ out,
    const int batch_size,
    const int in_features,
    const int out_features,
    const float multiplier,
    const float negative_slope
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_features;
    if (idx < total) {
        int row = idx / out_features;
        int col = idx % out_features;
        float val = 0.0f;
        for (int k = 0; k < in_features; k++) {
            val += x[row * in_features + k] * w[col * in_features + k];
        }
        val += bias[col];
        val *= multiplier;
        if (val < 0.0f) {
            val = val * negative_slope;
        }
        out[idx] = val;
    }
}

torch::Tensor fused_gemm_mul_leakyrelu_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor bias,
    float multiplier,
    float negative_slope
) {
    TORCH_CHECK(x.dim() == 2, "Input x must be 2D");
    TORCH_CHECK(w.dim() == 2, "Weight w must be 2D");
    TORCH_CHECK(bias.dim() == 1, "Bias must be 1D");
    TORCH_CHECK(x.size(1) == w.size(1), "Incompatible shapes for x and w");

    auto B = x.size(0);
    auto M = x.size(1);
    auto N = w.size(0);

    auto out = torch::empty({B, N}, x.options());
    int block_size = 256;
    int grid_size = (B * N + block_size - 1) / block_size;

    fused_gemm_mul_leakyrelu_kernel<<<grid_size, block_size>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        B,
        M,
        N,
        multiplier,
        negative_slope
    );
    return out;
}
"""

cpp_src = r"""
torch::Tensor fused_gemm_mul_leakyrelu_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor bias,
    float multiplier,
    float negative_slope
);
"""

fused_ops = load_inline(
    name="fused_gemm_mul_leakyrelu",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["fused_gemm_mul_leakyrelu_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    """
    Optimized model that fuses GEMM, multiply, and LeakyReLU in a custom CUDA kernel.
    """
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)  # for parameters
        self.multiplier = multiplier
        self.negative_slope = negative_slope

    def forward(self, x):
        return fused_ops.fused_gemm_mul_leakyrelu_cuda(
            x,
            self.gemm.weight,
            self.gemm.bias,
            self.multiplier,
            self.negative_slope
        )

batch_size = 1024
in_features  = 8192
out_features = 8192
multiplier = 2.0
negative_slope = 0.1

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, multiplier, negative_slope]
