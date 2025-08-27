import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------------
# CUDA / C++ source for fused ReLU + BiasAdd kernel
#   Reference computation: y = ReLU(x) + bias
# ---------------------------------------------------------------------------
source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

template <typename scalar_t>
__global__ void relu_bias_forward_kernel(const scalar_t* __restrict__ input,
                                         const scalar_t* __restrict__ bias,
                                         scalar_t* __restrict__ output,
                                         int N, int C, int H, int W) {
    const int total = N * C * H * W;
    const int stride = blockDim.x * gridDim.x;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
        const int hw  = H * W;
        const int c   = (idx / hw) % C;

        scalar_t val      = input[idx];
        scalar_t relu_val = val > scalar_t(0) ? val : scalar_t(0);
        output[idx]       = relu_val + bias[c];
    }
}

torch::Tensor bias_relu_forward_cuda(torch::Tensor input, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(),  "bias must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(bias.is_contiguous(),  "bias must be contiguous");
    TORCH_CHECK(input.scalar_type() == bias.scalar_type(),
                "input and bias must have the same data type");

    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);
    const int total = N * C * H * W;

    auto output = torch::empty_like(input);

    const int threads = 256;
    int blocks = (total + threads - 1) / threads;
    blocks = std::min(blocks, 65535);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(),
                                        "relu_bias_forward_cuda", ([&] {
        relu_bias_forward_kernel<scalar_t>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                input.data_ptr<scalar_t>(),
                bias.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                N, C, H, W);
    }));

    return output;
}
"""

cpp_src = "torch::Tensor bias_relu_forward_cuda(torch::Tensor input, torch::Tensor bias);"

# Compile / load the fused kernel
bias_relu = load_inline(
    name="bias_relu",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["bias_relu_forward_cuda"],
    verbose=False,
)

# ---------------------------------------------------------------------------
# Optimised model that leverages the custom fused kernel
# ---------------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Optimised model:
      Conv2d  ->  fused (ReLU + BiasAdd) CUDA kernel
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv(x)
        x = bias_relu.bias_relu_forward_cuda(
            x.contiguous(),                           # feature map
            self.bias.view(-1).contiguous()           # (C,)
        )
        return x
