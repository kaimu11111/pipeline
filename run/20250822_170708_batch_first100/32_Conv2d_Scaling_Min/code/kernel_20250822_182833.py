import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_scale_min_kernel(const float* __restrict__ x,
                                       float* __restrict__ out,
                                       int N, int C, int H, int W,
                                       float scale_factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * H * W;
    if (idx < total) {
        int n = idx / (H * W);
        int rem = idx % (H * W);
        int h = rem / W;
        int w = rem % W;

        int base = ((n * C) * H + h) * W + w;
        float min_val = x[base] * scale_factor;

        for (int c = 1; c < C; c++) {
            float val = x[base + c * H * W] * scale_factor;
            if (val < min_val) {
                min_val = val;
            }
        }
        int out_base = ((n * 1) * H + h) * W + w;
        out[out_base] = min_val;
    }
}

torch::Tensor fused_scale_min_cuda(torch::Tensor x, float scale_factor) {
    auto N = x.size(0);
    auto C = x.size(1);
    auto H = x.size(2);
    auto W = x.size(3);

    auto out = torch::zeros({N, 1, H, W}, x.options());

    int total = N * H * W;
    const int block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;

    fused_scale_min_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, H, W,
        scale_factor
    );

    return out;
}
"""

cpp_src = r"""
torch::Tensor fused_scale_min_cuda(torch::Tensor x, float scale_factor);
"""

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["fused_scale_min_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model that performs the same conv -> scale -> min operations,
    with the scale & min fused into a custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv(x)
        x = fused_ops.fused_scale_min_cuda(x, self.scale_factor)
        return x

batch_size = 64
in_channels = 64
out_channels = 128
height = width = 256
kernel_size = 3
scale_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]
