import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------------
# CUDA kernel: bias-add + ReLU fusion for NCHW float32 tensors
# ---------------------------------------------------------------------------
cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

namespace {

template<int THREADS>
__global__ void bias_relu_kernel(const float* __restrict__ x,
                                 const float* __restrict__ bias,
                                 float* __restrict__ y,
                                 const int C, const int H, const int W,
                                 const int total_elems) {
    int idx = blockIdx.x * THREADS + threadIdx.x;
    for (int i = idx; i < total_elems; i += gridDim.x * THREADS) {
        int c = (i / (H * W)) % C;          // channel index
        float v = x[i] + bias[c];
        y[i] = v > 0.f ? v : 0.f;           // ReLU
    }
}

} // anonymous namespace

torch::Tensor bias_relu_forward(torch::Tensor x, torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda(),  "input must be CUDA");
    TORCH_CHECK(bias.is_cuda(), "bias must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "only float32 supported");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "only float32 bias supported");
    TORCH_CHECK(x.dim() == 4, "input must be 4-D NCHW");

    x = x.contiguous();
    bias = bias.contiguous();

    const int N = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const int total_elems = N * C * H * W;

    auto y = torch::empty_like(x);

    constexpr int threads = 256;
    int blocks = (total_elems + threads - 1) / threads;
    blocks = std::min(blocks, 65535);   // CUDA block limit

    bias_relu_kernel<threads><<<blocks, threads>>>(
        x.data_ptr<float>(),
        bias.data_ptr<float>(),
        y.data_ptr<float>(),
        C, H, W,
        total_elems);

    return y;
}
"""

cpp_hdr = "torch::Tensor bias_relu_forward(torch::Tensor x, torch::Tensor bias);"

fused_ops = load_inline(
    name="bias_relu_fused",
    cpp_sources=cpp_hdr,
    cuda_sources=cuda_src,
    functions=["bias_relu_forward"],
    verbose=False,
)

# ---------------------------------------------------------------------------
# Optimised model using the fused CUDA kernel
# ---------------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Optimised model that keeps the PyTorch convolution but replaces
    bias-addition + ReLU with a single fused CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # learnable bias tensor (shape [C, 1, 1] or [C])
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv(x)
        x = fused_ops.bias_relu_forward(x, self.bias)
        return x

# ---------------------------------------------------------------------------
# Input helpers (same signatures as original for compatibility)
# ---------------------------------------------------------------------------
batch_size = 32
in_channels  = 32
out_channels = 64
height = width = 64
kernel_size = 3
bias_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]
