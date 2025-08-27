import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# --------------------------------------------------------------------
# CUDA kernel: fused ReLU + per-channel bias addition
# --------------------------------------------------------------------
cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x)  TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel: out = ReLU(in) + bias[channel]
__global__ void fused_relu_bias_kernel(const float* __restrict__ in,
                                       const float* __restrict__ bias,
                                       float* __restrict__ out,
                                       const int B, const int C,
                                       const int H, const int W) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = B * C * H * W;
    if (idx >= total) return;

    const int HW  = H * W;
    const int c   = (idx / HW) % C;      // channel index
    float val     = in[idx];
    val           = val > 0.f ? val : 0.f;  // ReLU
    out[idx]      = val + bias[c];          // add bias
}

torch::Tensor fused_relu_bias_forward(torch::Tensor input,
                                      torch::Tensor bias) {
    CHECK_INPUT(input);
    CHECK_INPUT(bias);

    const int B = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);

    auto output = torch::empty_like(input);

    const int total   = input.numel();
    const int threads = 256;
    const int blocks  = (total + threads - 1) / threads;

    fused_relu_bias_kernel<<<blocks, threads>>>(input.data_ptr<float>(),
                                                bias.data_ptr<float>(),
                                                output.data_ptr<float>(),
                                                B, C, H, W);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_relu_bias_forward", &fused_relu_bias_forward,
          "Fused ReLU + bias addition (CUDA)");
}
"""

cpp_hdr = "torch::Tensor fused_relu_bias_forward(torch::Tensor input, torch::Tensor bias);"

# Build/load the CUDA extension
fused_relu_bias = load_inline(name="fused_relu_bias",
                              cpp_sources=cpp_hdr,
                              cuda_sources=cuda_src,
                              functions=["fused_relu_bias_forward"],
                              verbose=False)

# --------------------------------------------------------------------
# Optimised model using the fused kernel
# --------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Optimised model: standard Conv2d followed by a fused CUDA kernel that
    performs ReLU and per-channel bias addition in a single pass.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size).cuda()
        self.bias = nn.Parameter(torch.randn(bias_shape, device="cuda"))
        self.fused_relu_bias = fused_relu_bias

    def forward(self, x):
        x = self.conv(x)
        # Kernel expects contiguous tensors
        x = x.contiguous()
        bias_flat = self.bias.view(-1).contiguous()  # (C)
        x = self.fused_relu_bias.fused_relu_bias_forward(x, bias_flat)
        return x

# --------------------------------------------------------------------
# Helper functions for external use
# --------------------------------------------------------------------
batch_size   = 32
in_channels  = 32
out_channels = 64
height = width = 64
kernel_size  = 3
bias_shape   = (out_channels, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device="cuda")]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]
