# <complete ModelNew code>
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


###############################################################################
# CUDA kernel: (bias + x) * scale followed by sigmoid, NCHW layout
###############################################################################
cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__device__ __forceinline__ scalar_t sigmoidf(scalar_t x) {
    return static_cast<scalar_t>(1.f) / (static_cast<scalar_t>(1.f) + expf(-x));
}

template <typename scalar_t>
__global__ void fused_bias_scale_sigmoid_kernel(
        const scalar_t* __restrict__ x,
        const scalar_t* __restrict__ bias,
        const scalar_t* __restrict__ scale,
        scalar_t* __restrict__ out,
        int C, int H, int W, int numel) {

    const int hw = H * W;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < numel;
         idx += blockDim.x * gridDim.x) {

        // Recover channel index from flattened idx (NCHW layout)
        int c = (idx / hw) % C;
        scalar_t val = x[idx];
        val = (val + bias[c]) * scale[c];
        out[idx] = sigmoidf(val);
    }
}

torch::Tensor fused_bias_scale_sigmoid_cuda(torch::Tensor x,
                                            torch::Tensor bias,
                                            torch::Tensor scale) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(scale.is_cuda(), "Scale must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Only float32 supported");

    auto N = x.size(0);
    auto C = x.size(1);
    auto H = x.size(2);
    auto W = x.size(3);
    auto numel = x.numel();

    auto out = torch::empty_like(x);

    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;

    fused_bias_scale_sigmoid_kernel<float><<<blocks, threads>>>(
        x.data_ptr<float>(),
        bias.data_ptr<float>(),
        scale.data_ptr<float>(),
        out.data_ptr<float>(),
        C, H, W, numel);

    return out;
}
"""

cpp_decls = """
torch::Tensor fused_bias_scale_sigmoid_cuda(torch::Tensor x,
                                            torch::Tensor bias,
                                            torch::Tensor scale);
"""

# Compile & load the CUDA extension
fused_bias_scale_sigmoid = load_inline(
    name="fused_bias_scale_sigmoid",
    cpp_sources=cpp_decls,
    cuda_sources=cuda_src,
    functions=["fused_bias_scale_sigmoid_cuda"],
    verbose=False,
)

###############################################################################
# Optimised model using the custom kernel
###############################################################################
class ModelNew(nn.Module):
    """
    Optimised model: convolution -> fused (bias + scale + sigmoid) CUDA kernel
    -> group normalization
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 num_groups,
                 bias_shape,
                 scale_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        # register parameters the same way as the original model
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = fused_bias_scale_sigmoid.fused_bias_scale_sigmoid_cuda(
            x,
            self.bias,
            self.scale
        )
        x = self.group_norm(x)
        return x


###############################################################################
# Helper functions (unchanged)
###############################################################################
batch_size = 64
in_channels = 4
out_channels = 16
height = width = 128
kernel_size = 3
num_groups = 8
bias_shape = (out_channels, 1, 1)
scale_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device="cuda")]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape]
# </complete ModelNew code>
