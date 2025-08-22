import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA source for fused activation + bias kernel
source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float gelu_fwd(float x) {
    // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    const float kBeta = float(0.70710678); // 1/sqrt(2)
    return 0.5f * x * (1.0f + erff(x * kBeta));
}

__global__ void fused_activation_bias_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int ncdhw,
    int C,
    int D,
    int H,
    int W,
    float negative_slope
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < ncdhw) {
        // Decompose idx into n, c, d, h, w
        int n = idx / (C * D * H * W);
        int c = (idx / (D * H * W)) % C;
        int d = (idx / (H * W)) % D;
        int h = (idx / W) % H;
        int w = idx % W;

        float val = input[idx];

        // ReLU
        val = fmaxf(val, 0.0f);
        // LeakyReLU
        val = (val > 0.0f) ? val : negative_slope * val;
        // GELU
        val = gelu_fwd(val);
        // Sigmoid
        val = 1.0f / (1.0f + expf(-val));

        // Add bias (bias is indexed only by channel)
        val += bias[c];

        output[idx] = val;
    }
}

torch::Tensor fused_activation_bias_forward(
    torch::Tensor input,
    torch::Tensor bias,
    float negative_slope
) {
    // input: (N, C, D, H, W)
    // bias:  (C) flattened from (C,1,1,1)
    auto sizes = input.sizes();
    int N = sizes[0];
    int C = sizes[1];
    int D = sizes[2];
    int H = sizes[3];
    int W = sizes[4];
    auto output = torch::empty_like(input);

    int ncdhw = N * C * D * H * W;
    const int blockSize = 256;
    int gridSize = (ncdhw + blockSize - 1) / blockSize;

    fused_activation_bias_kernel<<<gridSize, blockSize>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        ncdhw,
        C,
        D,
        H,
        W,
        negative_slope
    );

    return output;
}
"""

# Declarations in C++ for the functions to be compiled
cpp_src = r"""
torch::Tensor fused_activation_bias_forward(
    torch::Tensor input,
    torch::Tensor bias,
    float negative_slope
);
"""

# Build the custom CUDA extension
fused_activation_bias = load_inline(
    name="fused_activation_bias",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["fused_activation_bias_forward"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model that uses a standard Conv3d but fuses ReLU, LeakyReLU, GELU, Sigmoid,
    and bias addition into a single custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.negative_slope = 0.01

    def forward(self, x):
        x = self.conv(x)
        # Flatten bias to (C) to match the custom kernel's expectation
        bias_flat = self.bias.view(-1)
        return fused_activation_bias.fused_activation_bias_forward(
            x, bias_flat, self.negative_slope
        )

def get_inputs():
    # Same shape as original
    batch_size = 64
    in_channels = 8
    depth, height, width = 32, 64, 64
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    # Same initial arguments as original
    in_channels = 8
    out_channels = 32
    kernel_size = 3
    bias_shape = (out_channels, 1, 1, 1)
    return [in_channels, out_channels, kernel_size, bias_shape]
