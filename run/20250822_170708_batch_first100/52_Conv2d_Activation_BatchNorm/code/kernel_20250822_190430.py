import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Fused activation: x * tanh(softplus(x))
# softplus(x) = log(1 + exp(x))
# We replace torch.multiply(torch.tanh(torch.nn.functional.softplus(x)), x) with a custom CUDA kernel.

source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// CUDA kernel for fused activation
__global__ void fused_activation_kernel(const float* __restrict__ inp, float* __restrict__ out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = inp[idx];
        float sp = logf(1.0f + expf(val)); // softplus
        float t = tanhf(sp);
        out[idx] = val * t;               // fused activation
    }
}

torch::Tensor fused_activation_cuda(torch::Tensor x) {
    auto out = torch::empty_like(x);
    int size = x.numel();

    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;
    fused_activation_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);
    return out;
}
"""

cpp_src = r"""
torch::Tensor fused_activation_cuda(torch::Tensor x);
"""

# Load the CUDA extension
fused_activation = load_inline(
    name="fused_activation",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["fused_activation_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[],
)

class ModelNew(nn.Module):
    """
    Optimized model that reuses the convolution and BatchNorm layers from PyTorch,
    but replaces the expensive combination of tanh(softplus(x)) * x with a custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)

    def forward(self, x):
        x = self.conv(x)
        x = fused_activation.fused_activation_cuda(x)
        x = self.bn(x)
        return x

batch_size = 64
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
