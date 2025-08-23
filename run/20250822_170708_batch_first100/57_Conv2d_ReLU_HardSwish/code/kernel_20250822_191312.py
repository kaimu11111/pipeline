import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_relu_hardswish_kernel(const float* __restrict__ in, float* __restrict__ out, const int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = in[idx];
        // ReLU
        float relu_val = fmaxf(val, 0.0f);
        // HardSwish factor
        float c = (relu_val + 3.0f) / 6.0f;
        c = fminf(fmaxf(c, 0.0f), 1.0f);
        // Multiply for HardSwish
        out[idx] = relu_val * c;
    }
}

torch::Tensor fused_relu_hardswish_cuda(torch::Tensor input)
{
    auto output = torch::zeros_like(input);
    int size = input.numel();

    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;

    fused_relu_hardswish_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );

    return output;
}
"""

cpp_src = r"""
torch::Tensor fused_relu_hardswish_cuda(torch::Tensor input);
"""

fused_relu_hardswish = load_inline(
    name="fused_relu_hardswish",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["fused_relu_hardswish_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    """
    Optimized model that uses a fused custom CUDA kernel for ReLU + HardSwish.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.fused_relu_hardswish = fused_relu_hardswish

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_relu_hardswish.fused_relu_hardswish_cuda(x)
        return x

def get_inputs():
    batch_size = 128
    in_channels = 8
    height, width = 128, 128
    return [torch.randn(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [8, 64, 3]
