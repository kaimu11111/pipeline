import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA source for ReLU
relu_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* __restrict__ inp, float* __restrict__ out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = inp[idx];
        out[idx] = val > 0.0f ? val : 0.0f;
    }
}

torch::Tensor relu_cuda(torch::Tensor input) {
    auto output = torch::zeros_like(input);
    int size = input.numel();

    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;

    relu_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );

    return output;
}
"""

# Corresponding C++ function declaration
relu_cpp_src = r"""
torch::Tensor relu_cuda(torch::Tensor input);
"""

# Build the inline extension
relu_ops = load_inline(
    name="relu_ops",
    cpp_sources=relu_cpp_src,
    cuda_sources=relu_source,
    functions=["relu_cuda"],
    verbose=False,
    extra_cflags=[],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    """
    Optimized model that uses a custom CUDA kernel for ReLU,
    surrounding the built-in ConvTranspose3d and GroupNorm layers.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels)
        self.relu_ops = relu_ops

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.relu_ops.relu_cuda(x)
        x = self.group_norm(x)
        return x

def get_inputs():
    batch_size = 16
    in_channels = 64
    D, H, W = 32, 32, 32
    return [torch.rand(batch_size, in_channels, D, H, W)]

def get_init_inputs():
    in_channels = 64
    out_channels = 128
    kernel_size = 3
    groups = 8
    bias = False
    return [in_channels, out_channels, kernel_size, groups, bias]
