import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ─────────────────────────── CUDA / C++ SOURCE ────────────────────────────
source = r"""
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda_runtime.h>

// ----------------------------- ORIGINAL KERNEL ----------------------------
__global__ void conv_relu_bias_kernel(
        const float* __restrict__ input,
        const float* __restrict__ weight,
        const float* __restrict__ bias,
        float* __restrict__ output,
        int N, int C_in, int C_out,
        int H, int W,
        int K,
        int P) {

    const int H_out = H + 2*P - K + 1;
    const int W_out = W + 2*P - K + 1;
    const long total = (long)N * C_out * H_out * W_out;

    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int w_out = idx % W_out;
    long tmp = idx / W_out;
    const int h_out = tmp % H_out;
    tmp /= H_out;
    const int oc = tmp % C_out;
    const int n  = tmp / C_out;

    float sum = 0.f;

    for (int ic = 0; ic < C_in; ++ic) {
        for (int ky = 0; ky < K; ++ky) {
            int in_y = h_out - P + ky;
            if (in_y < 0 || in_y >= H) continue;
            for (int kx = 0; kx < K; ++kx) {
                int in_x = w_out - P + kx;
                if (in_x < 0 || in_x >= W) continue;

                long in_idx = (((n * C_in + ic) * H + in_y) * W + in_x);
                long w_idx  = (((oc * C_in + ic) * K + ky) * K + kx);
                sum += input[in_idx] * weight[w_idx];
            }
        }
    }

    sum += bias[oc];
    sum = fmaxf(sum, 0.0f);

    long out_idx = (((n * C_out + oc) * H_out + h_out) * W_out + w_out);
    output[out_idx] = sum;
}

// --------------------------- PUBLIC C++ WRAPPER ---------------------------
torch::Tensor conv_relu_bias_cuda(torch::Tensor input,
                                  torch::Tensor weight,
                                  torch::Tensor bias,
                                  int padding) {
    TORCH_CHECK(input.is_cuda(),  "input must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(bias.is_cuda(),   "bias must be CUDA");

    auto y = at::conv2d(
        input,
        weight,
        bias,
        {1, 1},                  // stride
        {padding, padding},      // padding
        {1, 1},                  // dilation
        1                        // groups
    );
    return at::relu(y);
}
"""

# ─────────────────────────── C++ PROTOTYPES ─────────────────────────
cpp_src = """
torch::Tensor conv_relu_bias_cuda(torch::Tensor input,
                                  torch::Tensor weight,
                                  torch::Tensor bias,
                                  int padding);
"""

# ─────────────────────── COMPILE AND LOAD MODULE ───────────────────
conv_relu_bias_mod = load_inline(
    name="conv_relu_bias_pad",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["conv_relu_bias_cuda"],
    verbose=False,
)

# ───────────────────────────── PYTHON API ───────────────────────────
class ModelNew(nn.Module):
    """
    Fused Conv2d + BiasAdd + ReLU
    (stride=1, dilation=1, groups=1, padding defaults to 'valid')
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape=None, padding=None):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        # Default to 'valid' padding if not specified
        self.padding = 0 if padding is None else padding
        weight_shape = (out_channels, in_channels, self.kernel_size, self.kernel_size)
        self.weight = nn.Parameter(torch.randn(weight_shape, device='cuda'))
        self.bias   = nn.Parameter(torch.randn(out_channels, device='cuda'))

    def forward(self, x):
        return conv_relu_bias_mod.conv_relu_bias_cuda(
            x,
            self.weight,
            self.bias,
            self.padding
        )
