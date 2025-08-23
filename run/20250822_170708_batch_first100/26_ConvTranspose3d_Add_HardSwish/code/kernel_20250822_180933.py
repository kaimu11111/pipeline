import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Inline fused CUDA operator that adds two tensors and applies HardSwish in a single pass
fused_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

__global__ void fused_add_hardswish_kernel(
    const float* __restrict__ x,
    const float* __restrict__ add,
    float* __restrict__ out,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx] + add[idx];
        // HardSwish factor: val * ReLU6(val + 3) / 6
        float tmp = val + 3.0f;
        tmp = fminf(fmaxf(tmp, 0.0f), 6.0f);
        float hswish = val * (tmp / 6.0f);
        // x * HardSwish(x)
        out[idx] = val * hswish; 
    }
}

torch::Tensor fused_add_hardswish_cuda(torch::Tensor x, torch::Tensor add) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;

    fused_add_hardswish_kernel<<<grid_size, block_size>>>(
        x.data_ptr<float>(),
        add.data_ptr<float>(),
        out.data_ptr<float>(),
        size
    );
    return out;
}
"""

fused_cpp_src = r"""
torch::Tensor fused_add_hardswish_cuda(torch::Tensor x, torch::Tensor add);
"""

# Build the fused operator
fused_add_hardswish = load_inline(
    name="fused_add_hardswish",
    cpp_sources=[fused_cpp_src],
    cuda_sources=[fused_source],
    functions=["fused_add_hardswish_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D transposed convolution, then fuses addition and HardSwish in a single custom kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_add_hardswish = fused_add_hardswish

    def forward(self, x, add_input):
        x = self.conv_transpose(x)
        x = self.fused_add_hardswish.fused_add_hardswish_cuda(x, add_input)
        return x
