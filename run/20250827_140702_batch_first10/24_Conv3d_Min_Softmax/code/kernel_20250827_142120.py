import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ----------------------------------------------------------------------
# CUDA kernels (min-reduction over depth  +  channel-wise softmax)
# ----------------------------------------------------------------------
cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

/////////////////////////////////////////////////////////////////
//  Reduce-min along depth dimension (dim==2)
//  Input : (N, C, D, H, W)  ->  Output : (N, C, H, W)
/////////////////////////////////////////////////////////////////
__global__ void min_depth_kernel(const float* __restrict__ inp,
                                 float* __restrict__ out,
                                 int N, int C, int D, int H, int W) {
    int idx  = blockIdx.x * blockDim.x + threadIdx.x;
    int size = N * C * H * W;
    if (idx >= size) return;

    int w =  idx % W;
    int h = (idx / W)       % H;
    int c = (idx / (W*H))   % C;
    int n =  idx / (C*H*W);

    int base = (((n*C + c)*D + 0)*H + h)*W + w;
    float min_val = inp[base];

    for (int d = 1; d < D; ++d) {
        int off = (((n*C + c)*D + d)*H + h)*W + w;
        float v  = inp[off];
        if (v < min_val) min_val = v;
    }
    out[idx] = min_val;
}

/////////////////////////////////////////////////////////////////
//  Channel-wise softmax (dim==1)
//  Input/Output : (N, C, H, W)
/////////////////////////////////////////////////////////////////
__global__ void softmax_channel_kernel(const float* __restrict__ inp,
                                       float* __restrict__ out,
                                       int N, int C, int H, int W) {
    int idx  = blockIdx.x * blockDim.x + threadIdx.x;
    int size = N * H * W;
    if (idx >= size) return;

    int w =  idx % W;
    int h = (idx / W)     % H;
    int n =  idx / (H*W);

    int strideHW = H * W;
    int base = (n * C * strideHW) + h * W + w;  // channel 0 address

    // Find max for numerical stability
    float max_val = inp[base];
    for (int c = 1; c < C; ++c) {
        float v = inp[base + c * strideHW];
        if (v > max_val) max_val = v;
    }

    // Exponentiate & accumulate
    float sum_exp = 0.0f;
    for (int c = 0; c < C; ++c) {
        float val = expf(inp[base + c * strideHW] - max_val);
        out[base + c * strideHW] = val;   // store temp exp
        sum_exp += val;
    }

    // Normalize
    for (int c = 0; c < C; ++c) {
        out[base + c * strideHW] /= sum_exp;
    }
}

/////////////////////////////////////////////////////////////////
//  C++ – accessible wrappers
/////////////////////////////////////////////////////////////////
torch::Tensor min_reduce_depth_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32,
                "only float32 tensors are supported");

    input = input.contiguous();

    const int N = input.size(0);
    const int C = input.size(1);
    const int D = input.size(2);
    const int H = input.size(3);
    const int W = input.size(4);

    auto output = torch::empty({N, C, H, W}, input.options());

    const int threads = 256;
    const int blocks  = (N * C * H * W + threads - 1) / threads;

    min_depth_kernel<<<blocks, threads>>>(input.data_ptr<float>(),
                                          output.data_ptr<float>(),
                                          N, C, D, H, W);
    return output;
}

torch::Tensor softmax_channel_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32,
                "only float32 tensors are supported");

    input = input.contiguous();

    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);

    auto output = torch::empty_like(input);

    const int threads = 256;
    const int blocks  = (N * H * W + threads - 1) / threads;

    softmax_channel_kernel<<<blocks, threads>>>(input.data_ptr<float>(),
                                                output.data_ptr<float>(),
                                                N, C, H, W);
    return output;
}
"""

cpp_src = r"""
torch::Tensor min_reduce_depth_cuda(torch::Tensor input);
torch::Tensor softmax_channel_cuda(torch::Tensor input);
"""

# Compile and load kernels
kernels = load_inline(
    name="optim_kernels",
    cpp_sources=[cpp_src],
    cuda_sources=[cuda_src],
    functions=["min_reduce_depth_cuda", "softmax_channel_cuda"],
    verbose=False,
)

# ----------------------------------------------------------------------
# Optimised model
# ----------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Optimised version of the original model.
    Uses:
        • PyTorch Conv3d
        • Custom CUDA kernel for min-reduction over depth
        • Custom CUDA kernel for channel-wise softmax
    """
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size).cuda()
        self.dim  = dim  # expecting 2 for depth reduction
        self.kernels = kernels

    def forward(self, x):
        x = self.conv(x)
        # Custom min over depth when applicable
        if self.dim == 2 and x.is_cuda and x.dtype == torch.float32:
            x = self.kernels.min_reduce_depth_cuda(x)
        else:
            x = torch.min(x, dim=self.dim)[0]
        # Channel-wise softmax via custom kernel
        x = self.kernels.softmax_channel_cuda(x)
        return x

# ----------------------------------------------------------------------
# Helper functions (same signature as original)
# ----------------------------------------------------------------------
batch_size = 64
in_channels = 3
out_channels = 12
D, H, W = 12, 16, 16
kernel_size = 3
dim = 2

def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W, device='cuda')]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, dim]
