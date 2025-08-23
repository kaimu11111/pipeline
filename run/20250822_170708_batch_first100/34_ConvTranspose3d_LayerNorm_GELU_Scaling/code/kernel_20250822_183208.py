import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Fused GELU + Scaling CUDA kernel
source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// Approximate GELU as in PyTorch: x * 0.5 * [1 + tanh(sqrt(2/pi)*(x + 0.044715x^3))]
__global__ void gelu_scale_kernel(const float* __restrict__ in, float* __restrict__ out, int size, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = in[idx];
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        out[idx] = x * cdf * scale;
    }
}

torch::Tensor gelu_scale_cuda(torch::Tensor input, float scale) {
    auto out = torch::zeros_like(input);
    int size = input.numel();
    const int blockSize = 256;
    const int gridSize = (size + blockSize - 1) / blockSize;

    gelu_scale_kernel<<<gridSize, blockSize>>>(
        input.data_ptr<float>(),
        out.data_ptr<float>(),
        size,
        scale
    );

    return out;
}
"""

cpp_src = r"""
torch::Tensor gelu_scale_cuda(torch::Tensor input, float scale);
"""

# Build and load the custom CUDA extension
gelu_scale = load_inline(
    name="gelu_scale",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["gelu_scale_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that reuses ConvTranspose3d and LayerNorm from PyTorch, 
    then applies a fused GELU+scale operation on GPU.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, eps=1e-5, scaling_factor=1.0):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, 
                                                 stride=stride, padding=padding, bias=bias)
        self.layer_norm = nn.LayerNorm(out_channels, eps=eps)
        self.scaling_factor = scaling_factor

        # Expose the fused operator
        self.gelu_scale = gelu_scale

    def forward(self, x):
        x = self.conv_transpose(x)               # (N, out_channels, D', H', W')
        x = self.layer_norm(x)                   # layer norm
        x = self.gelu_scale.gelu_scale_cuda(x, self.scaling_factor)  # fused GELU + scale
        return x

# Keep the same input generation for testing
batch_size = 32
in_channels = 32
out_channels = 64
D, H, W = 16, 32, 32
kernel_size = 4
stride = 2
padding = 1
bias = True
eps = 1e-5
scaling_factor = 1.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias, eps, scaling_factor]
