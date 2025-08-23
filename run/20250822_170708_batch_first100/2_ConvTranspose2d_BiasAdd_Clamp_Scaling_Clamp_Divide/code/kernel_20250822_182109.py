import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA source
source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float clamp01(float x) {
    return fmaxf(0.0f, fminf(x, 1.0f));
}

__global__ void fused_ops_kernel(const float* __restrict__ x_in,
                                 const float* __restrict__ bias,
                                 float* __restrict__ x_out,
                                 int N, int C, int H, int W,
                                 float scaling_factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx < total) {
        int c = (idx / (H * W)) % C;
        float val = x_in[idx] + bias[c];
        val = clamp01(val);
        val *= scaling_factor;
        val = clamp01(val);
        val /= scaling_factor;
        x_out[idx] = val;
    }
}

torch::Tensor fused_ops_cuda(torch::Tensor x,
                             torch::Tensor bias,
                             double scaling_factor) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    auto x_contig = x.contiguous();
    auto bias_contig = bias.view({-1}).contiguous();

    int N = x_contig.size(0);
    int C = x_contig.size(1);
    int H = x_contig.size(2);
    int W = x_contig.size(3);

    auto out = torch::zeros_like(x_contig);

    int total = N * C * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    fused_ops_kernel<<<blocks, threads>>>(x_contig.data_ptr<float>(),
                                          bias_contig.data_ptr<float>(),
                                          out.data_ptr<float>(),
                                          N, C, H, W,
                                          (float)scaling_factor);

    return out;
}
"""

# C++ declaration
cpp_src = r"""
torch::Tensor fused_ops_cuda(torch::Tensor x,
                             torch::Tensor bias,
                             double scaling_factor);
"""

# Build/load the fused operators
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["fused_ops_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    """
    Optimized Model that keeps the ConvTranspose2d from PyTorch but fuses bias addition,
    clamp, scaling and division into a single custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                                 stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_ops.fused_ops_cuda(x, self.bias, self.scaling_factor)
        return x
