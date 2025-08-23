import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

batch_size = 16
in_channels = 32
out_channels = 64
D, H, W = 32, 64, 64
kernel_size = 3

# Custom CUDA code for fused Mish + Tanh
source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_mish_tanh_kernel(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float sp = logf(1.0f + expf(x));   // softplus
        float mish_val = x * tanhf(sp);    // mish
        output[idx] = tanhf(mish_val);     // tanh(mish(x))
    }
}

torch::Tensor fused_mish_tanh_cuda(torch::Tensor input) {
    auto out = torch::zeros_like(input);
    int size = input.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    fused_mish_tanh_kernel<<<blocks, threads>>>(input.data_ptr<float>(),
                                               out.data_ptr<float>(),
                                               size);
    return out;
}
""";

cpp_src = r"""
torch::Tensor fused_mish_tanh_cuda(torch::Tensor input);
""";

fused_mish_tanh = load_inline(
    name="fused_mish_tanh",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["fused_mish_tanh_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    """
    Optimized Model that uses a fused Mish+Tanh custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.conv(x)
        x = fused_mish_tanh.fused_mish_tanh_cuda(x)
        return x

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
