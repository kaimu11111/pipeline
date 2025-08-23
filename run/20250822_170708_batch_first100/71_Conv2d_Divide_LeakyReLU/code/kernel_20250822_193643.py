import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom kernel to fuse division and LeakyReLU
source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void div_leaky_relu_kernel(const float* __restrict__ input,
                                      float* __restrict__ output,
                                      const float divisor,
                                      const float neg_slope,
                                      const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx] / divisor;
        if (val < 0) {
            val = val * neg_slope;
        }
        output[idx] = val;
    }
}

torch::Tensor div_leaky_relu_cuda(torch::Tensor input, float divisor, float neg_slope) {
    auto output = torch::zeros_like(input);
    int size = input.numel();

    const int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    div_leaky_relu_kernel<<<grid_size, block_size>>>(input.data_ptr<float>(),
                                                     output.data_ptr<float>(),
                                                     divisor,
                                                     neg_slope,
                                                     size);
    return output;
}
"""

cpp_src = r"""
torch::Tensor div_leaky_relu_cuda(torch::Tensor input, float divisor, float neg_slope);
"""

div_leaky_relu = load_inline(
    name="div_leaky_relu",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["div_leaky_relu_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    """
    Optimized model that fuses division and LeakyReLU into a custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = float(divisor)
        self.neg_slope = 0.01

    def forward(self, x):
        x = self.conv(x)
        x = div_leaky_relu.div_leaky_relu_cuda(x, self.divisor, self.neg_slope)
        return x

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3
divisor = 2

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divisor]
