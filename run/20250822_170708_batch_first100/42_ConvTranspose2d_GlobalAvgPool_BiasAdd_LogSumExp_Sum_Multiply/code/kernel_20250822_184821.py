import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Inline CUDA/C++ sources for custom kernels
source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// Kernel 1: Global average pooling + bias addition
__global__ void global_avg_add_bias_kernel(
    const float* __restrict__ inp,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int N, int C, int H, int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * C) {
        int n = idx / C;
        int c = idx % C;
        // Sum across H, W
        float sum_val = 0.0f;
        int hw_size = H * W;
        // Offset for this (n, c) slice
        int offset_in = (n * C + c) * H * W;
        for (int i = 0; i < hw_size; i++) {
            sum_val += inp[offset_in + i];
        }
        sum_val /= (float)hw_size; // Global average
        // Add bias
        sum_val += bias[c];
        // Write to output [n, c, 0, 0]
        out[idx] = sum_val;
    }
}

torch::Tensor global_avg_add_bias(torch::Tensor input, torch::Tensor bias) {
    TORCH_CHECK(input.dim() == 4, "Input must be 4D: (N, C, H, W)");
    TORCH_CHECK(bias.dim() == 3 || bias.dim() == 1,
                "Bias must be either shape (C) or (C,1,1)");
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);

    // Flatten bias if necessary
    auto bias_flat = bias;
    if (bias_flat.dim() == 3) {
        TORCH_CHECK(bias_flat.size(0) == C,
                    "Bias dimension mismatch in channels");
        bias_flat = bias_flat.view({C});
    }

    // Create an output tensor shaped [N, C, 1, 1], but we'll flatten the last two dims
    // for convenience. We'll store it as NxC in memory, then reshape back in Python.
    auto out = torch::empty({N, C}, input.options());

    const int block_size = 256;
    const int grid_size = (N * C + block_size - 1) / block_size;

    global_avg_add_bias_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        bias_flat.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, H, W
    );

    return out.reshape({N, C, 1, 1});
}

// Kernel 2: Log-sum-exp over channel dimension (dim=1) for a shape [N, C, 1, 1]
__global__ void logsumexp_channels_kernel(
    const float* __restrict__ inp,
    float* __restrict__ out,
    int N, int C)
{
    // Each block handles one or more "N" items, ignoring the 1,1 dims
    // We'll launch N threads, each responsible for one batch index
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N) {
        // We'll do log-sum-exp across the channel dimension
        float max_val = -FLT_MAX;
        int base_index = n * C; // ignoring the 1,1 for indexing in 2D flattened
        // find max
        for (int c = 0; c < C; c++) {
            float val = inp[base_index + c];
            if (val > max_val) {
                max_val = val;
            }
        }
        // sum exp
        float sum_exp = 0.0f;
        for (int c = 0; c < C; c++) {
            sum_exp += expf(inp[base_index + c] - max_val);
        }
        // final log-sum-exp
        float lse = max_val + logf(sum_exp);
        // store at [n, 0, 0, 0]
        out[n] = lse;
    }
}

torch::Tensor logsumexp_channels(torch::Tensor input) {
    TORCH_CHECK(input.dim() == 4, "Input must be 4D: (N, C, 1, 1) expected");
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    // We expect 1,1 for dims 2 and 3

    // Output shape will be [N, 1, 1, 1], but we'll flatten to Nx1 in a temporary buffer
    auto out = torch::empty({N}, input.options());

    const int block_size = 256;
    const int grid_size = (N + block_size - 1) / block_size;

    logsumexp_channels_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C
    );

    // Reshape to [N, 1, 1, 1]
    return out.reshape({N, 1, 1, 1});
}

// Kernel 3: Sum across (2, 3) for shape [N, 1, 1, 1] -> [N, 1] and multiply by 10
__global__ void sum_2_3_and_mul_kernel(
    const float* __restrict__ inp,
    float* __restrict__ out,
    int N)
{
    // Launch N threads
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N) {
        // shape is [N,1,1,1], sum across dims (2,3) is trivial = inp[n,0,0,0]
        float val = inp[n];
        // multiply by 10
        out[n] = val * 10.0f;
    }
}

torch::Tensor sum_2_3_and_mul(torch::Tensor input) {
    // shape [N, 1, 1, 1], we'll treat it as [N] in memory
    TORCH_CHECK(input.dim() == 4, "Input must be 4D: (N, 1, 1, 1) expected");
    int64_t N = input.size(0);

    // Flatten the input to [N] for kernel convenience
    auto inp_flat = input.view({N});
    auto out = torch::empty({N}, input.options());

    const int block_size = 256;
    const int grid_size = (N + block_size - 1) / block_size;

    sum_2_3_and_mul_kernel<<<grid_size, block_size>>>(
        inp_flat.data_ptr<float>(),
        out.data_ptr<float>(),
        N
    );

    // Final shape is [N, 1]
    return out.reshape({N, 1});
}
"""

cpp_src = r"""
torch::Tensor global_avg_add_bias(torch::Tensor input, torch::Tensor bias);
torch::Tensor logsumexp_channels(torch::Tensor input);
torch::Tensor sum_2_3_and_mul(torch::Tensor input);
"""

# Load the inline extension with the above custom kernels
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=[
        "global_avg_add_bias",
        "logsumexp_channels",
        "sum_2_3_and_mul"
    ],
    verbose=False
)

class ModelNew(nn.Module):
    """
    Optimized model that offloads global average pooling, bias addition, log-sum-exp, sum, 
    and multiplication to custom CUDA kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        # Keep PyTorch ConvTranspose2d for demonstration, can be replaced if desired
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_ops = fused_ops

    def forward(self, x):
        # 1) Transposed convolution (PyTorch)
        x = self.conv_transpose(x)
        # 2) Fused custom CUDA kernel for global average + bias
        x = self.fused_ops.global_avg_add_bias(x, self.bias)
        # 3) Custom log-sum-exp across channels
        x = self.fused_ops.logsumexp_channels(x)
        # 4) Sum across (2,3) and multiply by 10
        x = self.fused_ops.sum_2_3_and_mul(x)
        return x
