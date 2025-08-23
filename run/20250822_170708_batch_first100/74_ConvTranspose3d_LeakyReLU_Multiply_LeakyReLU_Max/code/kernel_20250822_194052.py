import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA code for fused LeakyReLU -> Multiply -> LeakyReLU
source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_lrelu_mult_lrelu_kernel(
    const float* x, 
    const float* multiplier, 
    float* out, 
    const int size, 
    const float neg_slope
)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        // First LeakyReLU
        if (val < 0.0f) {
            val = neg_slope * val;
        }
        // Multiply
        val *= multiplier[idx];
        // Second LeakyReLU
        if (val < 0.0f) {
            val = neg_slope * val;
        }
        out[idx] = val;
    }
}

torch::Tensor fused_lrelu_mult_lrelu_cuda(
    torch::Tensor x, 
    torch::Tensor multiplier, 
    float neg_slope
) {
    auto out = torch::zeros_like(x);

    int size = x.numel();
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;

    fused_lrelu_mult_lrelu_kernel<<<grid_size, block_size>>>(
        x.data_ptr<float>(),
        multiplier.data_ptr<float>(),
        out.data_ptr<float>(),
        size,
        neg_slope
    );

    return out;
}
"""

cpp_src = r"""
torch::Tensor fused_lrelu_mult_lrelu_cuda(torch::Tensor x, torch::Tensor multiplier, float neg_slope);
"""

# Compile the inline CUDA code
fused_lrelu_mult_lrelu = load_inline(
    name="fused_lrelu_mult_lrelu",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["fused_lrelu_mult_lrelu_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a 3D transposed convolution, followed by a custom fused
    LeakyReLU->Multiply->LeakyReLU CUDA kernel, then a max pooling operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=stride, 
            padding=padding, 
            output_padding=output_padding
        )
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.max_pool = nn.MaxPool3d(kernel_size=2)
        self.neg_slope = 0.2

    def forward(self, x):
        # 1) Transposed Convolution
        x = self.conv_transpose(x)
        # 2) Fused LeakyReLU -> Multiply -> LeakyReLU
        #    We expand the multiplier to match x's shape for proper elementwise broadcasting
        multiplier_expanded = self.multiplier.expand_as(x)
        x = fused_lrelu_mult_lrelu.fused_lrelu_mult_lrelu_cuda(x, multiplier_expanded, self.neg_slope)
        # 3) Max Pool
        x = self.max_pool(x)
        return x

batch_size = 16
in_channels = 16
out_channels = 32
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
multiplier_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape]
