import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Inline CUDA code defining two custom kernels: one for scaling and one for global average pooling 3D
source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// Kernel to multiply each element by a scale factor
__global__ void scale_kernel(const float* __restrict__ inp,
                             const float scale,
                             float* __restrict__ out,
                             const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = inp[idx] * scale;
    }
}

// Kernel to perform global average pooling on a 5D tensor [N, C, D, H, W]
// Produces output with shape [N, C, 1, 1, 1]
__global__ void global_avg_pool3d_kernel(const float* __restrict__ inp,
                                         float* __restrict__ out,
                                         const int N,
                                         const int C,
                                         const int D,
                                         const int H,
                                         const int W) {
    // Each (blockIdx.x) represents a unique (N, C) pair
    // out has dimensions [N, C, 1, 1, 1], so the linear index is blockIdx.x
    int nc = blockIdx.x;
    if (nc < N * C) {
        int n = nc / C;
        int c = nc % C;

        // Compute sum over D*H*W for input[n, c, ...]
        float val = 0.0f;
        int plane_size = D * H * W;
        int offset = n * C * plane_size + c * plane_size;

        for (int i = 0; i < plane_size; i++) {
            val += inp[offset + i];
        }
        val /= static_cast<float>(plane_size);

        // Store mean in output[n, c, 0, 0, 0]
        out[nc] = val;
    }
}

torch::Tensor scale_cuda(torch::Tensor input, float scale) {
    auto out = torch::empty_like(input);
    int size = input.numel();

    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;
    scale_kernel<<<grid_size, block_size>>>(input.data_ptr<float>(), scale, out.data_ptr<float>(), size);

    return out;
}

torch::Tensor global_avg_pool3d_cuda(torch::Tensor input) {
    // input is [N, C, D, H, W], output should be [N, C, 1, 1, 1]
    auto sizes = input.sizes();
    int N = sizes[0];
    int C = sizes[1];
    int D = sizes[2];
    int H = sizes[3];
    int W = sizes[4];

    // Output shape = [N, C, 1, 1, 1]
    auto out = torch::empty({N, C, 1, 1, 1}, input.options());

    const int block_size = 256;
    const int grid_size = (N * C + block_size - 1) / block_size;
    global_avg_pool3d_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, D, H, W
    );

    return out;
}
""";

# C++ function declarations
cpp_src = r"""
torch::Tensor scale_cuda(torch::Tensor input, float scale);
torch::Tensor global_avg_pool3d_cuda(torch::Tensor input);
""";

# Build the custom CUDA extension
custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_src,
    cuda_sources=source,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
    functions=["scale_cuda", "global_avg_pool3d_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    """
    Optimized Model that replaces the scale and global average pooling operations with custom CUDA kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.scale_factor = scale_factor
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size)
        self.batch_norm = nn.BatchNorm3d(out_channels, eps=eps, momentum=momentum)

    def forward(self, x):
        # Regular transpose conv
        x = self.conv_transpose(x)
        # Custom scale kernel
        x = custom_ops.scale_cuda(x, self.scale_factor)
        # BatchNorm
        x = self.batch_norm(x)
        # Custom global avg pool kernel
        x = custom_ops.global_avg_pool3d_cuda(x)
        return x
