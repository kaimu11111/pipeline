import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom fused kernel for adding bias, scaling, and applying sigmoid in one pass
source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_bias_scale_sigmoid_kernel(
    const float* __restrict__ x,
    const float* __restrict__ bias,
    const float* __restrict__ scale,
    float* __restrict__ out,
    const int batch_size,
    const int channels,
    const int height,
    const int width)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * channels * height * width;

    if (idx < total_size) {
        int w = idx % width;
        int h = (idx / width) % height;
        int c = (idx / (width * height)) % channels;
        // int n = idx / (width * height * channels); // not needed if we don't do anything with 'n'

        // input index
        float val = x[idx];

        // Apply bias and scale
        val += bias[c];
        val *= scale[c];
        
        // Sigmoid
        val = 1.0f / (1.0f + expf(-val));

        out[idx] = val;
    }
}

torch::Tensor fused_bias_scale_sigmoid_cuda(
    torch::Tensor x,
    torch::Tensor bias,
    torch::Tensor scale)
{
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(scale.is_cuda(), "scale must be a CUDA tensor");

    auto out = torch::empty_like(x);

    int batch_size = x.size(0);
    int channels   = x.size(1);
    int height     = x.size(2);
    int width      = x.size(3);
    int total_size = batch_size * channels * height * width;

    const int block_size = 256;
    const int grid_size = (total_size + block_size - 1) / block_size;

    fused_bias_scale_sigmoid_kernel<<<grid_size, block_size>>>(
        x.data_ptr<float>(),
        bias.data_ptr<float>(),
        scale.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        channels,
        height,
        width
    );

    return out;
}
"""

cpp_src = r"""
torch::Tensor fused_bias_scale_sigmoid_cuda(
    torch::Tensor x,
    torch::Tensor bias,
    torch::Tensor scale);
"""

# Compile the inline CUDA code for the fused op
fused_op = load_inline(
    name="fused_bias_scale_sigmoid",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["fused_bias_scale_sigmoid_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a convolution, then a single fused kernel for bias addition,
    scaling, sigmoid, and then group normalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape, device='cuda'))
        self.scale = nn.Parameter(torch.randn(scale_shape, device='cuda'))
        self.group_norm = nn.GroupNorm(num_groups, out_channels).cuda()

    def forward(self, x):
        x = self.conv(x)
        x = fused_op.fused_bias_scale_sigmoid_cuda(x, self.bias, self.scale)
        x = self.group_norm(x)
        return x
