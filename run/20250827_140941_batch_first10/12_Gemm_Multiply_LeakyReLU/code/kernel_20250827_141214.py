import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------------
# CUDA kernel + interface
# ---------------------------------------------------------------------------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void scale_leaky_relu_kernel(
        const scalar_t* __restrict__ input,
        scalar_t* __restrict__ output,
        const scalar_t multiplier,
        const scalar_t negative_slope,
        const int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        scalar_t val = input[idx] * multiplier;
        output[idx] = val > scalar_t(0) ? val : val * negative_slope;
    }
}

torch::Tensor scale_leaky_relu_cuda(torch::Tensor input,
                                    const double multiplier,
                                    const double negative_slope) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    auto output = torch::empty_like(input);

    const int size = input.numel();
    const int threads = 256;
    const int blocks  = (size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(),
                                        "scale_leaky_relu_cuda",
                                        ([&] {
        const scalar_t mul   = static_cast<scalar_t>(multiplier);
        const scalar_t slope = static_cast<scalar_t>(negative_slope);

        scale_leaky_relu_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            mul,
            slope,
            size);
    }));

    return output;
}
"""

cpp_src = """
torch::Tensor scale_leaky_relu_cuda(torch::Tensor input,
                                    const double multiplier,
                                    const double negative_slope);
"""

scale_leaky_relu = load_inline(
    name="scale_leaky_relu",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["scale_leaky_relu_cuda"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)

# ---------------------------------------------------------------------------
# Python autograd wrapper
# ---------------------------------------------------------------------------

class _ScaleLeakyReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, multiplier: float, negative_slope: float):
        ctx.save_for_backward(input)
        ctx.multiplier = float(multiplier)
        ctx.negative_slope = float(negative_slope)
        return scale_leaky_relu.scale_leaky_relu_cuda(
            input.contiguous(), ctx.multiplier, ctx.negative_slope
        )

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        multiplier = ctx.multiplier
        negative_slope = ctx.negative_slope
        grad_input = torch.where(
            input > 0,
            grad_output * multiplier,
            grad_output * multiplier * negative_slope,
        )
        # multiplier and negative_slope are constants, so no gradients for them
        return grad_input, None, None


def fused_scale_leaky_relu(input, multiplier: float, negative_slope: float):
    return _ScaleLeakyReLUFunction.apply(input, multiplier, negative_slope)

# ---------------------------------------------------------------------------
# Optimised Model
# ---------------------------------------------------------------------------

class ModelNew(nn.Module):
    """
    Optimised model that uses a fused CUDA kernel for
    (x * multiplier) followed by LeakyReLU.
    """
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.register_buffer('_multiplier', torch.tensor(float(multiplier)))
        self.register_buffer('_negative_slope', torch.tensor(float(negative_slope)))

    def forward(self, x):
        x = self.gemm(x)
        x = fused_scale_leaky_relu(
            x,
            float(self._multiplier.item()),
            float(self._negative_slope.item()),
        )
        return x

# ---------------------------------------------------------------------------
# Helpers to match original API
# ---------------------------------------------------------------------------

batch_size = 512
in_features  = 4096
out_features = 4096
multiplier = 2.0
negative_slope = 0.1

def get_inputs():
    return [torch.rand(batch_size, in_features, device='cuda')]

def get_init_inputs():
    return [in_features, out_features, multiplier, negative_slope]
