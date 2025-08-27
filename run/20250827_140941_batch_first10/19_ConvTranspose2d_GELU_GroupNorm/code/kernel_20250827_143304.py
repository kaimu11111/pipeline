import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------
# Custom CUDA implementation of GELU (approximation) – element-wise
# ---------------------------------------------------------------------
cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__device__ __forceinline__ scalar_t gelu_device(const scalar_t x) {
    const scalar_t kAlpha = 0.7978845608028654;          // sqrt(2/pi)
    return static_cast<scalar_t>(0.5) * x *
           (static_cast<scalar_t>(1.0) +
            tanh(kAlpha * (x + static_cast<scalar_t>(0.044715) * x * x * x)));
}

template <typename scalar_t>
__global__ void gelu_forward_kernel(const scalar_t* __restrict__ in,
                                    scalar_t* __restrict__ out,
                                    size_t numel) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        out[idx] = gelu_device(in[idx]);
    }
}

torch::Tensor gelu_forward_cuda(torch::Tensor input) {
    const size_t numel = input.numel();
    auto output = torch::empty_like(input);

    const int threads = 256;
    const int blocks  = (numel + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "gelu_forward_cuda", ([&] {
            gelu_forward_kernel<scalar_t><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                numel);
        }));

    return output;
}
"""

cpp_src = "torch::Tensor gelu_forward_cuda(torch::Tensor input);"

gelu_mod = load_inline(
    name="custom_gelu_cuda",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["gelu_forward_cuda"],
    verbose=False,
)

# ---------------------------------------------------------------------
# Optimised model using the custom CUDA GELU
# ---------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution, applies a custom-CUDA GELU,
    and normalizes with GroupNorm.
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, groups, num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride
        )
        self.group_norm = nn.GroupNorm(num_groups=num_groups,
                                       num_channels=out_channels)
        self._gelu = gelu_mod.gelu_forward_cuda

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self._gelu(x)
        x = self.group_norm(x)
        return x
