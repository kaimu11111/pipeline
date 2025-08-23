import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA/C++ source code for custom swish and hardswish kernels
source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void swish_kernel(const float* __restrict__ inp,
                             float* __restrict__ out,
                             const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = inp[idx];
        out[idx] = x * sigmoidf(x);
    }
}

__global__ void hardswish_kernel(const float* __restrict__ inp,
                                 float* __restrict__ out,
                                 const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = inp[idx];
        // hardswish: x * relu6(x+3) / 6
        float relu6_val = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
        out[idx] = x * (relu6_val / 6.0f);
    }
}

torch::Tensor swish_cuda(torch::Tensor input) {
    auto out = torch::zeros_like(input);
    int size = input.numel();
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;
    swish_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        out.data_ptr<float>(),
        size
    );
    return out;
}

torch::Tensor hardswish_cuda(torch::Tensor input) {
    auto out = torch::zeros_like(input);
    int size = input.numel();
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;
    hardswish_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        out.data_ptr<float>(),
        size
    );
    return out;
}
'''

cpp_declarations = r'''
torch::Tensor swish_cuda(torch::Tensor input);
torch::Tensor hardswish_cuda(torch::Tensor input);
'''

# Load our inline kernels
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=cpp_declarations,
    cuda_sources=source,
    functions=["swish_cuda", "hardswish_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    """
    Optimized model that uses custom CUDA kernels for Swish and HardSwish,
    while keeping PyTorch's built-in 3D transposed convolution and GroupNorm.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias
        )
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)

    def forward(self, x):
        # 3D transposed convolution
        x = self.conv_transpose(x)
        # Swish activation (custom)
        x = fused_ops.swish_cuda(x)
        # Group normalization (standard PyTorch)
        x = self.group_norm(x)
        # HardSwish activation (custom)
        x = fused_ops.hardswish_cuda(x)
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
groups = 4
eps = 1e-5

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, groups, eps]
