import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA/C++ source for fused LogSumExp + HardSwish + Subtract + Clamp
cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel to fuse logsumexp over channel dim=1, then apply HardSwish, subtract a single bias, and clamp
__global__ void fused_logsumexp_hardswish_sub_clamp_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int D, int H, int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElems = N * D * H * W;
    if (idx >= totalElems) return;

    // Convert linear index to (n, d, h, w)
    int w_i = idx % W;
    int tmp = idx / W;
    int h_i = tmp % H;
    tmp = tmp / H;
    int d_i = tmp % D;
    int n_i = tmp / D;

    // First pass: find max
    float max_val = -1.0e30f;
    int base_idx = ((n_i * C) * D + d_i) * H + h_i;
    base_idx = base_idx * W + w_i; // now base_idx is the index for channel=0
    int step_c = D * H * W;        // stride to go to next channel

    // Reduce max over channels
    for(int c = 0; c < C; c++){
        float val = input[base_idx + c * step_c];
        if(val > max_val) {
            max_val = val;
        }
    }

    // Second pass: sum of exp
    float sum_exp = 0.0f;
    for(int c = 0; c < C; c++){
        float val = input[base_idx + c * step_c];
        sum_exp += expf(val - max_val);
    }

    // logsumexp
    float lse = max_val + logf(sum_exp);

    // HardSwish: x * sigmoid(x+3) / 6
    // Here, "lse" plays the role of x
    float hs = lse *  (1.0f / (1.0f + expf(-(lse + 3.0f)))) / 6.0f;

    // subtract bias (only 1 element)
    float out_val = hs - bias[0];

    // clamp to [-1, 1]
    if(out_val < -1.0f) out_val = -1.0f;
    if(out_val > 1.0f)  out_val =  1.0f;

    // Store in output at channel=0
    // output shape: [N, 1, D, H, W]
    // linear index for output is the same as for the "no channel" dimension
    int out_idx = ((n_i * 1) * D + d_i) * H + h_i;
    out_idx = out_idx * W + w_i;
    output[out_idx] = out_val;
}

torch::Tensor fused_logsumexp_hardswish_sub_clamp_cuda(
    torch::Tensor x,
    torch::Tensor bias)
{
    TORCH_CHECK(x.dim() == 5, "Input must be 5D [N, C, D, H, W]");
    TORCH_CHECK(bias.numel() == 1, "Bias must have 1 element");
    int N = x.size(0);
    int C = x.size(1);
    int D = x.size(2);
    int H = x.size(3);
    int W = x.size(4);

    // Allocate output with shape [N, 1, D, H, W]
    auto out_sizes = std::vector<int64_t>{N, 1, D, H, W};
    auto out = torch::zeros(out_sizes, x.options());

    // Launch kernel
    int totalElems = N * D * H * W;
    int blockSize = 256;
    int gridSize = (totalElems + blockSize - 1) / blockSize;

    fused_logsumexp_hardswish_sub_clamp_kernel<<<gridSize, blockSize>>>(
        x.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, D, H, W
    );
    return out;
}
""";

# C++ declaration
cpp_declaration = r"""
torch::Tensor fused_logsumexp_hardswish_sub_clamp_cuda(
    torch::Tensor x,
    torch::Tensor bias);
"""

# Load/fuse custom operator
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=cpp_declaration,
    cuda_sources=cuda_src,
    functions=["fused_logsumexp_hardswish_sub_clamp_cuda"],
    verbose=False
)


class ModelNew(nn.Module):
    """
    Optimized model that retains ConvTranspose3d from PyTorch and replaces the subsequent
    logsumexp + HardSwish + bias-sub + clamp with a fused custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        # Reuse the same bias param (single element) as the original
        self.bias = nn.Parameter(torch.randn(1, 1, 1, 1))

    def forward(self, x):
        # Run built-in transposed convolution
        x = self.conv_transpose(x)
        # Apply the fused CUDA kernel
        x = fused_ops.fused_logsumexp_hardswish_sub_clamp_cuda(x, self.bias)
        return x
