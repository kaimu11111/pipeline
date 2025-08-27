# 1. Imports
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# 2. source – CUDA kernel + host wrapper
source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

/* ----------------------------------------------
   Fused bias-add + residual + mul + residual
   Forward : y = ((x + bias) + x) * x + x
           = (2*x + bias) * x + x
   ----------------------------------------------*/
template <typename scalar_t>
__global__ void fused_forward_kernel(const scalar_t* __restrict__ x,
                                     const scalar_t* __restrict__ bias,
                                     scalar_t* __restrict__ y,
                                     const int  N,
                                     const int  C,
                                     const int  inner) {
    const int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * C * inner;
    if (idx >= total) return;

    const int c   = (idx / inner) % C;     // channel index
    const scalar_t b   = bias[c];
    const scalar_t val = x[idx];

    y[idx] = (static_cast<scalar_t>(2) * val + b) * val + val;
}

torch::Tensor fused_forward(torch::Tensor x, torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda() && bias.is_cuda(), "Tensors must be CUDA");
    TORCH_CHECK(x.is_contiguous() && bias.is_contiguous(),
                "Tensors must be contiguous");

    const int64_t N     = x.size(0);
    const int64_t C     = x.size(1);
    const int64_t inner = x.size(2) * x.size(3) * x.size(4);     // D*H*W
    const int64_t total = N * C * inner;

    auto y = torch::empty_like(x);

    const int threads = 256;
    const int blocks  = (total + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        x.scalar_type(), "fused_forward_kernel", ([&] {
            fused_forward_kernel<scalar_t><<<blocks, threads>>>(
                x.data_ptr<scalar_t>(),
                bias.data_ptr<scalar_t>(),
                y.data_ptr<scalar_t>(),
                static_cast<int>(N),
                static_cast<int>(C),
                static_cast<int>(inner));
        }));

    return y;
}
"""

# 3. cpp_src – prototypes
cpp_src = "torch::Tensor fused_forward(torch::Tensor x, torch::Tensor bias);"

# 4. load_inline
fused_ops = load_inline(
    name="fused_bias_residual",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["fused_forward"],
    verbose=False,
)

# 5. class ModelNew
class ModelNew(nn.Module):
    """
    Optimised model that fuses:
      1) bias add
      2) residual add
      3) element-wise multiplication
      4) final residual add
    """
    class _FusedBiasResidualFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor, bias: torch.Tensor):
            ctx.save_for_backward(x.detach(), bias.detach())
            return fused_ops.fused_forward(x.contiguous(), bias.contiguous())

        @staticmethod
        def backward(ctx, grad_out):
            x_detached, bias_detached = ctx.saved_tensors
            # dy/dx = 4*x + bias + 1   (all detached to avoid higher-order grads)
            grad_input = grad_out * (4 * x_detached + bias_detached + 1.0)

            # dy/dbias = x
            axes = (0, 2, 3, 4)                                  # sum over N,D,H,W
            grad_bias = (grad_out * x_detached).sum(dim=axes).view_as(bias_detached)
            return grad_input, grad_bias

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, output_padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(*bias_shape))

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self._FusedBiasResidualFn.apply(x, self.bias)
        return x
