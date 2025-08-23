import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdio>

__global__ void clamp_and_div_kernel(const float* __restrict__ inp,
                                     float* __restrict__ out,
                                     int size,
                                     float min_value,
                                     float divisor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = inp[idx];
        if (val < min_value) {
            val = min_value;
        }
        val /= divisor;
        out[idx] = val;
    }
}

torch::Tensor clamp_and_div_cuda(torch::Tensor input, float min_value, float divisor) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");

    auto out = torch::zeros_like(input);
    int size = input.numel();

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    clamp_and_div_kernel<<<gridSize, blockSize>>>(
        input.data_ptr<float>(),
        out.data_ptr<float>(),
        size,
        min_value,
        divisor
    );

    return out;
}
"""

cpp_src = r"""
torch::Tensor clamp_and_div_cuda(torch::Tensor input, float min_value, float divisor);
"""

clamp_and_div = load_inline(
    name="clamp_and_div",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["clamp_and_div_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                                 stride=stride, padding=padding)
        self.min_value = min_value
        self.divisor = divisor

    def forward(self, x):
        x = self.conv_transpose(x)
        x = clamp_and_div.clamp_and_div_cuda(x, self.min_value, self.divisor)
        return x

def get_inputs():
    batch_size = 16
    in_channels = 64
    depth, height, width = 24, 48, 48
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]

def get_init_inputs():
    return [64, 128, 3, 2, 1, -1.0, 2.0]
