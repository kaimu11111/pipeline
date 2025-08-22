import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# --------------------------------------------------
# Inline CUDA/C++ code for custom operators
# --------------------------------------------------
source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// --------------------------------------------------
// 1) Elementwise division by a constant
// --------------------------------------------------
__global__ void custom_div_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  float divisor,
                                  int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] / divisor;
    }
}

torch::Tensor custom_div_cuda(torch::Tensor input, float divisor) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);
    const int blockSize = 256;
    const int gridSize = (size + blockSize - 1) / blockSize;
    custom_div_kernel<<<gridSize, blockSize>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        divisor,
        size
    );
    return output;
}

// --------------------------------------------------
// 2) Bias addition (broadcast bias of shape [C, 1, 1, 1] over x of shape [N, C, D, H, W])
// --------------------------------------------------
__global__ void custom_bias_add_kernel(const float* __restrict__ input,
                                       const float* __restrict__ bias,
                                       float* __restrict__ output,
                                       int N, int C, int D, int H, int W) {
    // 1D index for [N, C, D, H, W]
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = N * C * D * H * W;
    if (idx < totalSize) {
        // Decompose idx into n, c, d, h, w
        int w_idx = idx % W;               // w
        int tmp = idx / W;
        int h_idx = tmp % H;              // h
        tmp /= H;
        int d_idx = tmp % D;              // d
        tmp /= D;
        int c_idx = tmp % C;              // c
        int n_idx = tmp / C;              // n

        // bias is [C, 1, 1, 1], so we index bias by [c_idx, 0, 0, 0]
        int bias_idx = c_idx; 
        output[idx] = input[idx] + bias[bias_idx];
    }
}

torch::Tensor custom_bias_add_cuda(torch::Tensor input, torch::Tensor bias) {
    // input shape is [N, C, D, H, W]
    // bias shape is [C, 1, 1, 1]
    int N = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);

    auto output = torch::zeros_like(input);
    int totalSize = N * C * D * H * W;

    const int blockSize = 256;
    const int gridSize = (totalSize + blockSize - 1) / blockSize;

    custom_bias_add_kernel<<<gridSize, blockSize>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D, H, W
    );
    return output;
}

// --------------------------------------------------
// 3) Sum along dimension=1 for 5D tensor [N, C, D, H, W] -> output [N, D, H, W]
// --------------------------------------------------
__global__ void custom_sum_dim1_kernel(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       int N, int C, int D, int H, int W) {
    // We launch N*D*H*W threads. Each thread sums across C.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = N * D * H * W;
    if (idx < totalSize) {
        // Decompose idx into n, d, h, w
        int w_idx = idx % W;               
        int tmp = idx / W;
        int h_idx = tmp % H;              
        tmp /= H;
        int d_idx = tmp % D;              
        int n_idx = tmp / D;

        float sum_val = 0.0f;
        // Sum across channel dimension
        for (int c = 0; c < C; c++) {
            int inputIndex = (((n_idx * C + c) * D + d_idx) * H + h_idx) * W + w_idx;
            sum_val += input[inputIndex];
        }
        int outIndex = (((n_idx * D + d_idx) * H + h_idx) * W + w_idx);
        output[outIndex] = sum_val;
    }
}

torch::Tensor custom_sum_dim1_cuda(torch::Tensor input) {
    // input shape is [N, C, D, H, W]
    int N = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);

    // output shape is [N, D, H, W]
    auto out_sizes = std::vector<int64_t>{N, D, H, W};
    auto output = torch::zeros(out_sizes, input.options());

    int totalSize = N * D * H * W;
    const int blockSize = 256;
    const int gridSize = (totalSize + blockSize - 1) / blockSize;

    custom_sum_dim1_kernel<<<gridSize, blockSize>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D, H, W
    );
    return output;
}

"""

# Declare function signatures for the above inline CUDA/C++ routines
cpp_src = r"""
torch::Tensor custom_div_cuda(torch::Tensor input, float divisor);
torch::Tensor custom_bias_add_cuda(torch::Tensor input, torch::Tensor bias);
torch::Tensor custom_sum_dim1_cuda(torch::Tensor input);
"""

# Load/fuse all custom operators into a single extension
custom_ops = load_inline(
    name="custom_cuda_ops",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["custom_div_cuda", "custom_bias_add_cuda", "custom_sum_dim1_cuda"],
    verbose=False,
)

# --------------------------------------------------
# Optimized Model: ModelNew
# --------------------------------------------------
class ModelNew(nn.Module):
    """
    Optimized Model that replaces certain PyTorch ops with custom CUDA kernels:
      1) Division by a constant
      2) Bias addition
      3) Sum along dimension=1
    Other steps (3D conv, max pool, global avg pool) remain the same as the original.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.max_pool = nn.MaxPool3d(pool_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # register bias as a parameter
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim
        # Expose CUDA functions
        self.custom_div = custom_ops
        self.custom_bias_add = custom_ops
        self.custom_sum_dim = custom_ops

    def forward(self, x):
        # Original flow:
        #  1) 3D conv
        #  2) / divisor
        #  3) max pool
        #  4) global avg pool
        #  5) + bias
        #  6) sum along dim
        x = self.conv(x)
        x = self.custom_div.custom_div_cuda(x, self.divisor)
        x = self.max_pool(x)
        x = self.global_avg_pool(x)
        x = self.custom_bias_add.custom_bias_add_cuda(x, self.bias)
        # We'll handle only sum_dim=1 with our custom kernel
        # If sum_dim != 1, switch to PyTorch's standard sum
        if self.sum_dim == 1:
            x = self.custom_sum_dim.custom_sum_dim1_cuda(x)
        else:
            x = torch.sum(x, dim=self.sum_dim)
        return x
