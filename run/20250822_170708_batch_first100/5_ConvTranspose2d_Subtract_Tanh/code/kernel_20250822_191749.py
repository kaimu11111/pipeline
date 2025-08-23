import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom fused CUDA kernel for subtracting bias and applying tanh
source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void sub_bias_tanh_kernel(
    const float* __restrict__ x, 
    const float* __restrict__ bias, 
    float* __restrict__ out,
    int batch_size,
    int channels,
    int height,
    int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels * height * width;
    if (idx < total) {
        int c = (idx / (height * width)) % channels;
        float val = x[idx] - bias[c];
        out[idx] = tanhf(val);
    }
}

torch::Tensor sub_bias_tanh_cuda(torch::Tensor x, torch::Tensor bias) {
    auto batch_size = x.size(0);
    auto channels = x.size(1);
    auto height = x.size(2);
    auto width = x.size(3);

    auto out = torch::zeros_like(x);

    int total = batch_size * channels * height * width;
    const int block_size = 256;
    const int grid_size = (total + block_size - 1) / block_size;

    sub_bias_tanh_kernel<<<grid_size, block_size>>>(
        x.data_ptr<float>(),
        bias.data_ptr<float>(),
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
torch::Tensor sub_bias_tanh_cuda(torch::Tensor x, torch::Tensor bias);
"""

# Load the fused operator
sub_bias_tanh = load_inline(
    name="sub_bias_tanh",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["sub_bias_tanh_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model: uses PyTorch's ConvTranspose2d followed by a fused CUDA kernel
    that subtracts bias and applies tanh in one pass.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=stride, 
            padding=padding, 
            output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv_transpose(x)
        # Flatten bias to [out_channels] for indexing in the fused kernel
        fused_bias = self.bias.view(-1)
        x = sub_bias_tanh.sub_bias_tanh_cuda(x, fused_bias)
        return x

# Keep the same input generation functions
batch_size = 32
in_channels = 64
out_channels = 64
height = width = 256
kernel_size = 4
bias_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]
