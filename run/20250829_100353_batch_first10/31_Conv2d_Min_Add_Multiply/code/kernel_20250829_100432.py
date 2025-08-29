# <complete ModelNew code>
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# --------------------------------------------------------------------
# CUDA kernel: fuse min-with-constant + channel-bias add + scalar scale
# --------------------------------------------------------------------
cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fuse_kernel(const float* __restrict__ x,
                            const float* __restrict__ bias,
                            float* __restrict__ out,
                            const int64_t total_elements,
                            const int64_t channels,
                            const int64_t spatial_size,
                            const float  constant_val,
                            const float  scale) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    // Recover channel index from flattened NCHW index
    const int64_t c = (idx / spatial_size) % channels;

    float v = x[idx];
    v = v < constant_val ? v : constant_val;   // min with constant
    v += bias[c];                              // add per-channel bias
    v *= scale;                                // multiply by scalar
    out[idx] = v;
}

torch::Tensor fuse_min_bias_scale_cuda(torch::Tensor x,
                                       torch::Tensor bias,
                                       float constant_val,
                                       float scale) {
    TORCH_CHECK(x.is_cuda(),  "Input tensor must be CUDA");
    TORCH_CHECK(bias.is_cuda(), "Bias tensor must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Only float32 supported");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "Only float32 bias supported");

    x = x.contiguous();
    bias = bias.contiguous().view({-1});   // flatten bias (C)

    auto out = torch::empty_like(x);

    const int64_t total_elements = x.numel();
    const int64_t channels       = x.size(1);
    const int64_t spatial_size   = x.size(2) * x.size(3); // H * W

    const int threads = 256;
    const int blocks  = (total_elements + threads - 1) / threads;

    fuse_kernel<<<blocks, threads>>>(x.data_ptr<float>(),
                                     bias.data_ptr<float>(),
                                     out.data_ptr<float>(),
                                     total_elements,
                                     channels,
                                     spatial_size,
                                     constant_val,
                                     scale);

    return out;
}
"""

cpp_decl = """
torch::Tensor fuse_min_bias_scale_cuda(torch::Tensor x,
                                       torch::Tensor bias,
                                       float constant_val,
                                       float scale);
"""

# Compile / load the inline extension (done once at import time)
_fused_ops = load_inline(name="fuse_min_bias_scale",
                         cpp_sources=cpp_decl,
                         cuda_sources=cuda_src,
                         functions=["fuse_min_bias_scale_cuda"],
                         verbose=False)

# --------------------
# Optimised torch model
# --------------------
class ModelNew(nn.Module):
    """
    Optimised model that keeps the convolution in PyTorch but fuses the
    subsequent min(+constant) + bias add + scalar multiplication into a
    single custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size, constant_value,
                 bias_shape, scaling_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.register_buffer("constant_value", torch.tensor(float(constant_value)))
        self.register_buffer("scaling_factor", torch.tensor(float(scaling_factor)))
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv(x)
        x = _fused_ops.fuse_min_bias_scale_cuda(
            x,
            self.bias.view(-1),
            float(self.constant_value),
            float(self.scaling_factor)
        )
        return x
