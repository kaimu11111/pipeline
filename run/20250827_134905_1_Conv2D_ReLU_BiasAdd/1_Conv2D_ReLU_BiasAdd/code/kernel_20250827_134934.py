import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------
# Fused ReLU + per-channel bias add CUDA kernel
# ---------------------------------------------------------------------
cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void relu_bias_kernel(const scalar_t* __restrict__ input,
                                 const scalar_t* __restrict__ bias,
                                 scalar_t* __restrict__ output,
                                 int N, int C, int H, int W) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * C * H * W;
    if (idx >= total) return;

    // Channel index for the current linear index (NCHW layout)
    const int c = (idx / (H * W)) % C;

    scalar_t val = input[idx];
    val = val > scalar_t(0) ? val : scalar_t(0);  // ReLU
    val += bias[c];                               // Add per-channel bias
    output[idx] = val;
}

torch::Tensor relu_bias_cuda(torch::Tensor input, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(),  "bias  must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "input must be 4-D tensor NCHW");
    TORCH_CHECK(bias.dim()  == 3 && bias.size(1) == 1 && bias.size(2) == 1,
                "bias shape must be (C,1,1)");
    TORCH_CHECK(input.size(1) == bias.size(0),
                "channel dimension of input and bias must match");
    TORCH_CHECK(input.scalar_type() == bias.scalar_type(),
                "input and bias must have the same dtype");

    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);

    auto output = torch::empty_like(input);

    const int threads = 256;
    const int total   = input.numel();
    const int blocks  = (total + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "relu_bias_cuda", ([&] {
        relu_bias_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N, C, H, W);
    }));

    return output;
}
"""

cpp_src = "torch::Tensor relu_bias_cuda(torch::Tensor input, torch::Tensor bias);"

fused_relu_bias = load_inline(
    name="fused_relu_bias",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["relu_bias_cuda"],
    verbose=False,
)

# ---------------------------------------------------------------------
# Optimised model using the fused CUDA kernel
# ---------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Optimised model:
        Conv2d  ->  fused (ReLU + bias add)
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self._fused_relu_bias = fused_relu_bias.relu_bias_cuda

    def forward(self, x):
        x = self.conv(x)
        # Ensure contiguous layout for direct pointer access
        x = self._fused_relu_bias(x.contiguous(), self.bias.contiguous())
        return x
