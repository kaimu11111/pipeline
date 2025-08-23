import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA code for subtracting mean across spatial dimensions of a 5D tensor: (B, C, D, H, W)
# We'll compute the mean per (B, C) and then subtract it from each spatial element.
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void compute_spatial_mean_kernel(const float* __restrict__ x,
                                            float* __restrict__ mean,
                                            const int B, const int C,
                                            const int D, const int H, const int W) {
    // Each block corresponds to one (batch, channel) pair
    int bc = blockIdx.x;
    int b = bc / C;
    int c = bc % C;
    int volume = D * H * W;

    float sum = 0.0f;
    for (int idx = threadIdx.x; idx < volume; idx += blockDim.x) {
        int d = idx / (H * W);
        int hw = idx % (H * W);
        int hh = hw / W;
        int ww = hw % W;
        // Compute the flattened offset
        int offset = ((b * C + c) * D + d) * H * W + hh * W + ww;
        sum += x[offset];
    }

    // Reduce within the block
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write out mean
    if (tid == 0) {
        mean[bc] = sdata[0] / volume;
    }
}

__global__ void sub_spatial_mean_kernel(float* __restrict__ x,
                                        const float* __restrict__ mean,
                                        const int B, const int C,
                                        const int D, const int H, const int W) {
    int bc = blockIdx.x;
    int b = bc / C;
    int c = bc % C;
    int volume = D * H * W;
    float m = mean[bc];

    for (int idx = threadIdx.x; idx < volume; idx += blockDim.x) {
        int d = idx / (H * W);
        int hw = idx % (H * W);
        int hh = hw / W;
        int ww = hw % W;
        int offset = ((b * C + c) * D + d) * H * W + hh * W + ww;
        x[offset] -= m;
    }
}

torch::Tensor sub_mean_3d_cuda(torch::Tensor x) {
    auto B = x.size(0);
    auto C = x.size(1);
    auto D = x.size(2);
    auto H = x.size(3);
    auto W = x.size(4);

    // Allocate space for the per-(B, C) means
    auto mean = torch::empty({B * C}, x.options());

    const int block_size = 256;
    const int grid_size = B * C;

    // Kernel 1: Compute mean
    compute_spatial_mean_kernel<<<grid_size, block_size>>>(
        x.data_ptr<float>(),
        mean.data_ptr<float>(),
        B, C, D, H, W
    );

    // Kernel 2: Subtract mean
    sub_spatial_mean_kernel<<<grid_size, block_size>>>(
        x.data_ptr<float>(),
        mean.data_ptr<float>(),
        B, C, D, H, W
    );

    return x;
}
"""

cpp_decl = r"""
torch::Tensor sub_mean_3d_cuda(torch::Tensor x);
"""

# Build the custom CUDA extension
sub_mean_3d = load_inline(
    name="sub_mean_3d",
    cpp_sources=cpp_decl,
    cuda_sources=cuda_source,
    functions=["sub_mean_3d_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    """
    A 3D convolutional transpose layer followed by Batch Normalization
    and a custom CUDA kernel to subtract mean along spatial dimensions.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias
        )
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.sub_mean_3d = sub_mean_3d

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        x = self.sub_mean_3d.sub_mean_3d_cuda(x)
        return x

batch_size = 16
in_channels = 16
out_channels = 32
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]
