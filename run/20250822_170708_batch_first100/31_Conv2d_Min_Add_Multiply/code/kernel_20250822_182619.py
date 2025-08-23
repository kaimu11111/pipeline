import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Global definitions, matching the original architecture
batch_size = 128
in_channels = 64
out_channels = 128
height = width = 128
kernel_size = 3
constant_value = 0.5
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor]

# Custom CUDA kernel source to fuse min, bias-add, and scaling operations
cuda_src = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_min_add_scale_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int H, int W,
    float constant_value,
    float scaling_factor) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = N * C * H * W;
    if (idx < total_size) {
        int nc_hw = C * H * W;
        int n = idx / nc_hw;
        int tmp = idx % nc_hw;
        int c = tmp / (H * W);
        tmp = tmp % (H * W);
        int h = tmp / W;
        int w = tmp % W;

        float val = input[idx];
        float min_val = val < constant_value ? val : constant_value;
        float out_val = (min_val + bias[c]) * scaling_factor;
        output[idx] = out_val;
    }
}

torch::Tensor fused_min_add_scale_cuda(
    torch::Tensor input,
    torch::Tensor bias,
    float constant_value,
    float scaling_factor)
{
    auto output = torch::empty_like(input);

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    int total_size = N * C * H * W;
    const int block_size = 256;
    int grid_size = (total_size + block_size - 1) / block_size;

    fused_min_add_scale_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        constant_value,
        scaling_factor
    );

    return output;
}
'''

cpp_src = r'''
torch::Tensor fused_min_add_scale_cuda(
    torch::Tensor input,
    torch::Tensor bias,
    float constant_value,
    float scaling_factor
);
'''

# Load and compile the fused CUDA extension
fused_extension = load_inline(
    name="fused_ops",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["fused_min_add_scale_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model that leverages a fused custom CUDA kernel to perform:
    min with a constant, add bias, and multiply by scaling factor
    after a standard convolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size).cuda()
        self.constant_value = constant_value
        # Matches original bias shape but stored as a parameter
        self.bias = nn.Parameter(torch.randn(bias_shape).cuda())
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv(x)
        # Flatten bias so that index c matches exactly
        bias_flat = self.bias.view(-1)
        # Use the fused custom CUDA operator
        x = fused_extension.fused_min_add_scale_cuda(
            x.contiguous(),
            bias_flat.contiguous(),
            float(self.constant_value),
            float(self.scaling_factor)
        )
        return x
