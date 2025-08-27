import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

template <typename scalar_t>
__global__ void fused_bias_clamp_scale_kernel(const scalar_t* __restrict__ x,
                                              const scalar_t* __restrict__ bias,
                                              scalar_t* __restrict__ out,
                                              int64_t elements,
                                              int64_t C, int64_t HW,
                                              float scaling) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elements) return;

    int64_t c   = (idx / HW) % C;             // channel index
    scalar_t v  = x[idx] + bias[c];           // add bias

    v = v < 0.f ? 0.f : (v > 1.f ? 1.f : v);  // clamp [0,1]
    v *= scaling;                             // scale
    v = v < 0.f ? 0.f : (v > 1.f ? 1.f : v);  // clamp again
    v /= scaling;                             // divide

    out[idx] = v;
}

torch::Tensor fused_bias_clamp_scale(torch::Tensor x,
                                     torch::Tensor bias,
                                     double scaling) {
    TORCH_CHECK(x.is_cuda() && bias.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 &&
                bias.scalar_type() == torch::kFloat32,
                "only float32 supported");

    x   = x.contiguous();
    bias= bias.contiguous();
    auto out = torch::empty_like(x);

    const int64_t N  = x.size(0);
    const int64_t C  = x.size(1);
    const int64_t H  = x.size(2);
    const int64_t W  = x.size(3);
    const int64_t HW = H * W;
    const int64_t elements = x.numel();

    const int threads = 256;
    const int blocks  = (elements + threads - 1) / threads;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    fused_bias_clamp_scale_kernel<float><<<blocks, threads, 0, stream>>>(
        x.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        elements, C, HW,
        static_cast<float>(scaling));

    return out;
}
"""

cpp_src = """
torch::Tensor fused_bias_clamp_scale(torch::Tensor x,
                                     torch::Tensor bias,
                                     double scaling);
"""

fused_ops = load_inline(
    name="fused_bias_clamp_scale_mod",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["fused_bias_clamp_scale"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"]
)

class ModelNew(nn.Module):
    """
    Fused model performing bias add, clamp, scale, clamp, and divide
    via a single custom CUDA kernel after ConvTranspose2d.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, output_padding, bias_shape, scaling_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        # Create bias on CPU to align RNG streams, then let .cuda() move it.
        self.bias = nn.Parameter(torch.randn(bias_shape, dtype=torch.float32))
        self.scaling_factor = float(scaling_factor)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_ops.fused_bias_clamp_scale(x, self.bias, self.scaling_factor)
        return x
