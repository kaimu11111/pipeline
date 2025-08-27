import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


# ---------------------------------------------------------------------------
# CUDA kernel (fused  min -> sum -> GELU -> bias-add)
# ---------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float gelu_approx(float x) {
    // Approximate GELU as used in PyTorch (tanh formulation)
    const float k0 = 0.7978845608f;        // sqrt(2.0 / pi)
    const float k1 = 0.044715f;
    return 0.5f * x * (1.0f + tanhf(k0 * (x + k1 * x * x * x)));
}

__global__ void fused_kernel(const float* __restrict__ x,
                             const float* __restrict__ bias,
                             float* __restrict__ out,
                             int N, int C, int H, int W) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * W;
    if (idx >= total) return;

    const int n = idx / W;   // batch index
    const int w = idx % W;   // width index

    float sum_over_h = 0.0f;

    for (int h = 0; h < H; ++h) {
        float min_over_c = INFINITY;
        for (int c = 0; c < C; ++c) {
            const int offset = (((n * C + c) * H + h) * W) + w; // NCHW indexing
            const float val = x[offset];
            if (val < min_over_c) {
                min_over_c = val;
            }
        }
        sum_over_h += min_over_c;
    }

    float res = gelu_approx(sum_over_h) + bias[0];
    out[idx] = res;
}

torch::Tensor fused_forward_cuda(torch::Tensor x, torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(bias.numel() == 1, "Bias is expected to hold a single element");

    x = x.contiguous();
    bias = bias.contiguous();

    const int64_t N = x.size(0);
    const int64_t C = x.size(1);
    const int64_t H = x.size(2);
    const int64_t W = x.size(3);

    auto out = torch::empty({N, 1, 1, W}, x.options());

    const int threads = 256;
    const int blocks = (N * W + threads - 1) / threads;

    fused_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        (int)N, (int)C, (int)H, (int)W
    );

    return out;
}
"""

cpp_decls = "torch::Tensor fused_forward_cuda(torch::Tensor x, torch::Tensor bias);"

fused_ops = load_inline(
    name="fused_min_sum_gelu_bias",
    cpp_sources=cpp_decls,
    cuda_sources=cuda_source,
    functions=["fused_forward_cuda"],
    verbose=False,
)

# ---------------------------------------------------------------------------
# Optimised model
# ---------------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Optimised version of the original model.

    Operations after the ConvTranspose2d are fused into one custom CUDA kernel:
        1) torch.min along channel dimension
        2) torch.sum along height dimension
        3) GELU activation
        4) Bias addition
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, output_padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride, padding, output_padding
        )
        # Bias that will be added after the fused kernel; expect single element
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv_transpose(x)
        # fused custom CUDA kernel replaces: min->sum->gelu->bias
        x = fused_ops.fused_forward_cuda(x, self.bias)
        return x
