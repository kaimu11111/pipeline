import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------------
# CUDA kernel: fused element-wise add + HardSwish + final multiply
# ---------------------------------------------------------------------------
cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_add_hardswish_kernel(const float* __restrict__ a,
                                           const float* __restrict__ b,
                                           float* __restrict__ out,
                                           const int numel) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    float y      = a[idx] + b[idx];                         // add
    float relu6  = fminf(fmaxf(y + 3.0f, 0.0f), 6.0f);      // ReLU6(y + 3)
    float hswish = y * relu6 * (1.0f / 6.0f);               // HardSwish(y)
    out[idx]     = y * hswish;                              // final multiply
}

torch::Tensor fused_add_hardswish_cuda(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(a.sizes() == b.sizes(),      "Input sizes must match");
    TORCH_CHECK(a.scalar_type() == torch::kFloat32,
                "Only float32 tensors are supported");

    auto out   = torch::empty_like(a);
    int  numel = a.numel();

    const int threads = 256;
    const int blocks  = (numel + threads - 1) / threads;

    fused_add_hardswish_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        numel
    );

    return out;
}
"""

cpp_decl = "torch::Tensor fused_add_hardswish_cuda(torch::Tensor a, torch::Tensor b);"

fused_add_hswish = load_inline(
    name        = "fused_add_hswish",
    cpp_sources = cpp_decl,
    cuda_sources= cuda_src,
    functions   = ["fused_add_hardswish_cuda"],
    verbose     = False
)

# ---------------------------------------------------------------------------
# Optimised PyTorch module
# ---------------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Model that performs a 3D transposed convolution, adds an input tensor,
    and applies HardSwish activation, all fused in a custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, output_padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        # Keep bias for compatibility (unused in forward, mirrors original)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_op = fused_add_hswish

    def forward(self, x, add_input):
        x = self.conv_transpose(x)
        x = self.fused_op.fused_add_hardswish_cuda(x, add_input)
        return x


# ---------------------------------------------------------------------------
# Helper functions required by the submission environment
# ---------------------------------------------------------------------------
batch_size      = 64
in_channels     = 16
out_channels    = 32
D, H, W         = 8, 8, 8
kernel_size     = 3
stride          = 2
padding         = 1
output_padding  = 1
bias_shape      = (out_channels, 1, 1, 1, 1)

def get_inputs():
    return [
        torch.rand(batch_size, in_channels, D, H, W, device="cuda"),
        torch.rand(batch_size, out_channels, D*stride, H*stride, W*stride, device="cuda")
    ]

def get_init_inputs():
    return [
        in_channels, out_channels, kernel_size,
        stride, padding, output_padding, bias_shape
    ]
