import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Inline C++ header declarations for our custom CUDA ops.
cpp_src = r"""
torch::Tensor min_dim2_cuda(torch::Tensor x);
torch::Tensor softmax_dim1_cuda(torch::Tensor x);
"""

# Inline CUDA/C++ source containing our hand-written kernels.
source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// ------------------------------------
// Min along dimension=2 kernel
// Input shape: (N, C, D, H, W)
// Output shape: (N, C, H, W)
// ------------------------------------
__global__ void min_dim2_kernel(const float* __restrict__ x,
                                float* __restrict__ out,
                                int N, int C, int D, int H, int W) {
    // Each thread handles one (N, C, H, W) coordinate
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;

    int w = idx % W;
    int temp = idx / W;
    int h = temp % H;
    temp /= H;
    int c = temp % C;
    int n = temp / C;

    // Compute the index of the first element in dimension D
    int base_idx = ((n * C + c) * D) * H * W;
    // Initialize with first slice along D
    float m_val = x[base_idx + (0 * H + h) * W + w];
    // Find minimum along dimension D
    for (int d = 1; d < D; d++) {
        float val = x[base_idx + (d * H + h) * W + w];
        if (val < m_val) {
            m_val = val;
        }
    }
    // Write result
    out[idx] = m_val;
}

torch::Tensor min_dim2_cuda(torch::Tensor x) {
    TORCH_CHECK(x.dim() == 5, "Input must have 5 dimensions (N, C, D, H, W).");

    const auto N = x.size(0);
    const auto C = x.size(1);
    const auto D = x.size(2);
    const auto H = x.size(3);
    const auto W = x.size(4);

    // Allocate output
    auto out = torch::empty({N, C, H, W}, x.options());

    const int threads = 256;
    const int blocks = (N * C * H * W + threads - 1) / threads;

    min_dim2_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, D, H, W
    );

    return out;
}

// ------------------------------------
// Softmax along dimension=1 kernel
// Input shape: (N, C, H, W)
// Output shape: (N, C, H, W)
// ------------------------------------
__global__ void softmax_dim1_kernel(const float* __restrict__ in,
                                    float* __restrict__ out,
                                    int N, int C, int H, int W) {
    // Each thread handles one (N, H, W) "row" across C
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * H * W;
    if (idx >= total) return;

    int w = idx % W;
    int temp = idx / W;
    int h = temp % H;
    int n = temp / H;

    // Find max for stable softmax
    float max_val = in[((n * C) * H + h) * W + w];
    for (int c = 1; c < C; c++) {
        float val = in[(((n * C) + c) * H + h) * W + w];
        if (val > max_val) {
            max_val = val;
        }
    }

    // Compute sum of exp(...) after subtracting max
    float sum_val = 0.0f;
    for (int c = 0; c < C; c++) {
        float val = in[(((n * C) + c) * H + h) * W + w];
        sum_val += expf(val - max_val);
    }

    // Write normalized exponent
    for (int c = 0; c < C; c++) {
        float val = in[(((n * C) + c) * H + h) * W + w];
        out[(((n * C) + c) * H + h) * W + w] = expf(val - max_val) / sum_val;
    }
}

torch::Tensor softmax_dim1_cuda(torch::Tensor x) {
    TORCH_CHECK(x.dim() == 4, "Input must have 4 dimensions (N, C, H, W).");

    const auto N = x.size(0);
    const auto C = x.size(1);
    const auto H = x.size(2);
    const auto W = x.size(3);

    auto out = torch::empty_like(x);

    const int threads = 256;
    const int blocks = (N * H * W + threads - 1) / threads;

    softmax_dim1_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, H, W
    );

    return out;
}
""".strip()

# Build our custom CUDA extension.
custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["min_dim2_cuda", "softmax_dim1_cuda"],
    verbose=False
)

# Our optimized model that replaces the min and softmax ops with custom CUDA kernels.
class ModelNew(nn.Module):
    """
    Optimized model that still performs a 3D convolution,
    but uses custom CUDA kernels for min (dim=2) and softmax (dim=1).
    """
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.dim = dim  # We'll assume dim=2 for min
        self.custom_ops = custom_ops

    def forward(self, x):
        x = self.conv(x)
        # If self.dim == 2, we use our custom min kernel, else fallback to torch.min
        if self.dim == 2:
            x = self.custom_ops.min_dim2_cuda(x)
        else:
            x = torch.min(x, self.dim)[0]
        # Then apply our custom softmax along dim=1
        x = self.custom_ops.softmax_dim1_cuda(x)
        return x
