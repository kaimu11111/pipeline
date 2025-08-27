import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


# ----------------------------------------------------------------------
# CUDA kernel: fused   y = x * (1 + scale)
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void scale_residual_forward_kernel(const float* __restrict__ input,
                                              float* __restrict__ output,
                                              const float total,
                                              const int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] * total;
    }
}

torch::Tensor scale_residual_forward_cuda(torch::Tensor input, float total) {
    // Expect float32, contiguous, CUDA tensor
    input = input.contiguous();

    const int size = input.numel();
    auto output = torch::empty_like(input);

    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    scale_residual_forward_kernel<<<blocks, threads>>>(input.data_ptr<float>(),
                                                       output.data_ptr<float>(),
                                                       total,
                                                       size);

    return output;
}
"""

cpp_source = r"""
torch::Tensor scale_residual_forward_cuda(torch::Tensor input, float total);
"""

scale_residual_mod = load_inline(
    name="scale_residual",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["scale_residual_forward_cuda"],
    verbose=False,
)


# ----------------------------------------------------------------------
# Autograd wrapper to preserve original gradient semantics
# (grad_out * scaling_factor)
# ----------------------------------------------------------------------
class _ScaleResidualFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scaling_factor: float):
        ctx.scaling_factor = float(scaling_factor)
        total = 1.0 + ctx.scaling_factor  # fused coefficient
        return scale_residual_mod.scale_residual_forward_cuda(input, total)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scaling_factor, None


# ----------------------------------------------------------------------
# Optimised model
# ----------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Optimised model:
      y = Linear(x)
      out = y * scaling_factor + y (residual, no grad through residual)
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.scaling_factor = float(scaling_factor)

    def forward(self, x):
        x = self.linear(x)
        x = _ScaleResidualFunction.apply(x, self.scaling_factor)
        return x


# ----------------------------------------------------------------------
# Helpers to match original interface
# ----------------------------------------------------------------------
batch_size = 4096
in_features = 2048
out_features = 2048
scaling_factor = 0.5


def get_inputs():
    return [torch.rand(batch_size, in_features, device="cuda")]


def get_init_inputs():
    return [in_features, out_features, scaling_factor]
