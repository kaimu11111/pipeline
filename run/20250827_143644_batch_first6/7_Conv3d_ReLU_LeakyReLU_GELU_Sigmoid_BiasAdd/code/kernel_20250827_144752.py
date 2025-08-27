import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------
# CUDA source – kernel + host wrapper
# ---------------------------------------------------------------------
source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_act_bias_kernel(const float* __restrict__ inp,
                                      const float* __restrict__ bias,
                                      float* __restrict__ out,
                                      int channels,
                                      int inner_size,
                                      int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    float v = inp[idx];

    // ReLU
    v = v > 0.0f ? v : 0.0f;

    // LeakyReLU (negative_slope = 0.01)
    v = v > 0.0f ? v : 0.01f * v;

    // GELU (tanh approximation)
    const float k0 = 0.7978845608028654f;   // sqrt(2/pi)
    const float k1 = 0.044715f;
    float inner = k0 * (v + k1 * v * v * v);
    float cdf   = 0.5f * (1.0f + tanhf(inner));
    v = v * cdf;

    // Sigmoid
    v = 1.0f / (1.0f + expf(-v));

    // Add bias (broadcast over spatial dims)
    int c = (idx / inner_size) % channels;
    v += bias[c];

    out[idx] = v;
}

torch::Tensor fused_act_bias_cuda(torch::Tensor x, torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda(),  "x must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32,  "x must be float32");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "bias must be float32");

    auto x_contig    = x.contiguous();
    auto bias_contig = bias.contiguous();
    auto out         = torch::empty_like(x_contig);

    const int channels   = x_contig.size(1);
    const int depth      = x_contig.size(2);
    const int height     = x_contig.size(3);
    const int width      = x_contig.size(4);
    const int inner_size = depth * height * width;
    const int total      = x_contig.numel();

    const int block_size = 256;
    const int grid_size  = (total + block_size - 1) / block_size;

    fused_act_bias_kernel<<<grid_size, block_size>>>(
        x_contig.data_ptr<float>(),
        bias_contig.data_ptr<float>(),
        out.data_ptr<float>(),
        channels,
        inner_size,
        total
    );

    return out;
}
"""

# ---------------------------------------------------------------------
# C++ prototypes
# ---------------------------------------------------------------------
cpp_src = r"""
torch::Tensor fused_act_bias_cuda(torch::Tensor x, torch::Tensor bias);
"""

# ---------------------------------------------------------------------
# Build & load the extension
# ---------------------------------------------------------------------
fused_ops = load_inline(
    name="fused_act_bias",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["fused_act_bias_cuda"],
    verbose=False,
)

# ---------------------------------------------------------------------
# PyTorch module that calls the fused CUDA kernel
# ---------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        # Padding set to 0 to match reference implementation
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=0)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self._fused = fused_ops

    def forward(self, x):
        x = self.conv(x)
        bias_flat = self.bias.view(-1).contiguous().to(x.device)
        x = self._fused.fused_act_bias_cuda(x, bias_flat)
        return x
