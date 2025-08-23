import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA/C++ source for mean pooling over depth + bias add, and fused softmax+tanh+scale
source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Kernel to compute mean pooling over depth + bias add
// x: (B, C, D, H, W)
// bias: (1, C, 1, 1, 1)
// out: (B, C, 1, H, W)
__global__ void mean_bias_kernel(const float* __restrict__ x,
                                 const float* __restrict__ bias,
                                 float* __restrict__ out,
                                 int B, int C, int D, int H, int W) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * H * W;
    if (index < total) {
        int b = index / (C * H * W);
        int r = index % (C * H * W);
        int c = r / (H * W);
        r = r % (H * W);
        int h = r / W;
        int w = r % W;

        // Accumulate sum over D dimension
        float val = 0.0f;
        int base_idx = ((b * C + c) * D + 0) * H * W;
        for(int d = 0; d < D; d++){
            val += x[base_idx + d * H * W + h * W + w];
        }
        val = val / D;

        // Add bias
        val += bias[c];

        // Store to out
        out[index] = val;
    }
}

// Stage1: compute exp(x) in out and accumulate sums in sum_exp for softmax
// x/out shape: (B, C, 1, H, W)
// sum_exp shape: (B, H, W)
__global__ void softmax_tanh_scale_kernel_stage1(const float* __restrict__ x,
                                                 float* __restrict__ out,
                                                 float* __restrict__ sum_exp,
                                                 int B, int C, int H, int W) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * H * W;
    if (index < total) {
        int b = index / (C * H * W);
        int r = index % (C * H * W);
        int c = r / (H * W);
        int r2 = r % (H * W);
        int h = r2 / W;
        int w = r2 % W;

        float val = x[index];
        float exp_val = expf(val);

        out[index] = exp_val;
        // accumulate sum
        atomicAdd(&sum_exp[b * H * W + h * W + w], exp_val);
    }
}

// Stage2: out = (out / sum_exp), then tanh, then scale
// out: (B, C, 1, H, W)
// sum_exp: (B, H, W)
__global__ void softmax_tanh_scale_kernel_stage2(float* __restrict__ out,
                                                 const float* __restrict__ sum_exp,
                                                 float scale,
                                                 int B, int C, int H, int W) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * H * W;
    if (index < total) {
        int b = index / (C * H * W);
        int r = index % (C * H * W);
        int c = r / (H * W);
        int r2 = r % (H * W);
        int h = r2 / W;
        int w = r2 % W;

        float denom = sum_exp[b * H * W + h * W + w];
        float val = out[index] / denom;   // softmax
        val = tanhf(val);                 // tanh
        val *= scale;                     // scale
        out[index] = val;
    }
}

torch::Tensor mean_bias_cuda(torch::Tensor x, torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");

    auto sizes = x.sizes();  // B, C, D, H, W
    int B = sizes[0];
    int C = sizes[1];
    int D = sizes[2];
    int H = sizes[3];
    int W = sizes[4];

    // Output shape: (B, C, 1, H, W)
    auto out = torch::zeros({B, C, 1, H, W}, x.options());

    // Launch kernel
    int total = B * C * H * W;
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;
    mean_bias_kernel<<<gridSize, blockSize>>>(x.data_ptr<float>(),
                                              bias.data_ptr<float>(),
                                              out.data_ptr<float>(),
                                              B, C, D, H, W);
    return out;
}

torch::Tensor softmax_tanh_scale_cuda(torch::Tensor x, float scale) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    auto sizes = x.sizes();  // (B, C, 1, H, W)
    int B = sizes[0];
    int C = sizes[1];
    int H = sizes[3];
    int W = sizes[4];

    // We do softmax over C dimension
    // We'll do two-stage kernel
    // out is the same shape as x
    auto out = x.clone();
    // sum_exp shape: (B, H, W)
    auto sum_exp = torch::zeros({B, H, W}, x.options());

    // Stage 1: compute exp, accumulate sums
    {
        int total = B * C * H * W;
        int blockSize = 256;
        int gridSize = (total + blockSize - 1) / blockSize;
        softmax_tanh_scale_kernel_stage1<<<gridSize, blockSize>>>(
            x.data_ptr<float>(),
            out.data_ptr<float>(),
            sum_exp.data_ptr<float>(),
            B, C, H, W
        );
    }

    // Stage 2: finalize softmax -> tanh -> scale
    {
        int total = B * C * H * W;
        int blockSize = 256;
        int gridSize = (total + blockSize - 1) / blockSize;
        softmax_tanh_scale_kernel_stage2<<<gridSize, blockSize>>>(
            out.data_ptr<float>(),
            sum_exp.data_ptr<float>(),
            scale,
            B, C, H, W
        );
    }
    return out;
}
""";

# C++ declarations
cpp_src = r"""
torch::Tensor mean_bias_cuda(torch::Tensor x, torch::Tensor bias);
torch::Tensor softmax_tanh_scale_cuda(torch::Tensor x, float scale);
""";

# Build the custom kernels
kernels = load_inline(
    name="model_optimized_kernels",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["mean_bias_cuda", "softmax_tanh_scale_cuda"],
    verbose=False,
)

# The new optimized model
class ModelNew(nn.Module):
    """
    Model that performs a series of operations with custom CUDA kernels:
    1. Transposed 3D convolution (PyTorch)
    2. Mean pooling (across depth) + bias add (custom)
    3. Softmax (across channels) + tanh + scaling (custom)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1).cuda())
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv_transpose(x)  # (B, C, D, H, W)
        x = kernels.mean_bias_cuda(x, self.bias)    # (B, C, 1, H, W)
        x = kernels.softmax_tanh_scale_cuda(x, self.scaling_factor)  # (B, C, 1, H, W)
        return x

def get_inputs():
    # Return a random input tensor for the model
    batch_size = 16
    in_channels = 16
    depth = 32
    height = width = 128
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]

def get_init_inputs():
    # Return init arguments matching the model signature
    in_channels = 16
    out_channels = 64
    kernel_size = 3
    stride = 1
    padding = 1
    scaling_factor = 2.0
    return [in_channels, out_channels, kernel_size, stride, padding, scaling_factor]
