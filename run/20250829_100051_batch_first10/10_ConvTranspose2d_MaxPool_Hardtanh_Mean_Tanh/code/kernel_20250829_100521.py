# <complete ModelNew code>
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ----------------------------------------------------------------------
# CUDA kernel + host wrapper (WITHOUT the PYBIND11_MODULE block)
# ----------------------------------------------------------------------
source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

////////////////////////////////////////////////////////////////////////////////
// CUDA kernel
////////////////////////////////////////////////////////////////////////////////
__global__ void fused_pool_hardtanh_mean_tanh_kernel(
        const float* __restrict__ x,   // [N, C, H, W]
        float* __restrict__ out,       // [N, C, 1, 1]
        const int N, const int C,
        const int H, const int W,
        const float hard_min,
        const float hard_max) {

    // One (block) handles one (n,c) pair
    const int nc = blockIdx.x;
    if (nc >= N * C) return;

    const int n = nc / C;
    const int c = nc % C;

    const int pooled_H = H >> 1;   // H/2
    const int pooled_W = W >> 1;   // W/2
    const int pooled_size = pooled_H * pooled_W;

    // Strides for NCHW
    const int stride_n = C * H * W;
    const int stride_c = H * W;
    const int stride_h = W;

    // Base pointer for (n,c,0,0)
    const int base_idx = n * stride_n + c * stride_c;

    // Each thread accumulates partial sum
    float partial_sum = 0.0f;
    for (int idx = threadIdx.x; idx < pooled_size; idx += blockDim.x) {
        const int ph = idx / pooled_W;          // pooled height index
        const int pw = idx % pooled_W;          // pooled width index

        const int h0 = (ph << 1);               // 2*ph
        const int h1 = h0 + 1;
        const int w0 = (pw << 1);               // 2*pw
        const int w1 = w0 + 1;

        // Fetch 4 values for 2x2 window
        const int idx00 = base_idx + h0 * stride_h + w0;
        const int idx01 = base_idx + h0 * stride_h + w1;
        const int idx10 = base_idx + h1 * stride_h + w0;
        const int idx11 = base_idx + h1 * stride_h + w1;

        float v0 = x[idx00];
        float v1 = x[idx01];
        float v2 = x[idx10];
        float v3 = x[idx11];

        // MaxPool2d (2x2)
        float max_val = fmaxf(fmaxf(v0, v1), fmaxf(v2, v3));

        // HardTanh
        float clamped = fminf(fmaxf(max_val, hard_min), hard_max);

        partial_sum += clamped;
    }

    // ------------------------------------------------------------------
    // Parallel reduction within the block
    // ------------------------------------------------------------------
    extern __shared__ float shm[];
    shm[threadIdx.x] = partial_sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shm[threadIdx.x] += shm[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Thread 0 writes the final result: mean + tanh
    if (threadIdx.x == 0) {
        float mean_val = shm[0] / static_cast<float>(pooled_size);
        out[nc] = tanhf(mean_val);
    }
}

////////////////////////////////////////////////////////////////////////////////
// C++ / PyTorch binding (no PYBIND11_MODULE here – added automatically by
// torch.utils.cpp_extension::load_inline)
////////////////////////////////////////////////////////////////////////////////
torch::Tensor fused_pool_hardtanh_mean_tanh_cuda(
        torch::Tensor input,
        float hard_min,
        float hard_max) {

    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(input.scalar_type() == at::kFloat,
                "Only float32 tensors are supported");
    TORCH_CHECK(input.dim() == 4, "Input must be a 4-D tensor [N,C,H,W]");
    TORCH_CHECK((input.size(2) & 1) == 0 && (input.size(3) & 1) == 0,
                "H and W must be even for 2×2 pooling");

    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);

    auto out = torch::empty({N, C, 1, 1}, input.options());

    const int threads = 256;
    const int blocks  = N * C;
    const int sh_mem  = threads * sizeof(float);

    fused_pool_hardtanh_mean_tanh_kernel<<<blocks, threads, sh_mem>>>(
        input.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, H, W,
        hard_min,
        hard_max);

    return out;
}
"""

# ----------------------------------------------------------------------
# C++ prototypes exposed to Python
# ----------------------------------------------------------------------
cpp_src = """
torch::Tensor fused_pool_hardtanh_mean_tanh_cuda(torch::Tensor input,
                                                 float hard_min,
                                                 float hard_max);
"""

# ----------------------------------------------------------------------
# Build / load the extension
# ----------------------------------------------------------------------
fused_ops = load_inline(
    name="fused_pool_hardtanh_mean_tanh",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["fused_pool_hardtanh_mean_tanh_cuda"],
    verbose=False,
)

# ----------------------------------------------------------------------
# PyTorch module that uses the fused CUDA kernel
# ----------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Optimised model that keeps the ConvTranspose2d layer from the original
    architecture but fuses MaxPool2d → HardTanh → Mean → Tanh into a single
    custom CUDA kernel.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 maxpool_kernel_size,
                 maxpool_stride,
                 hardtanh_min,
                 hardtanh_max):
        super().__init__()

        # Original ConvTranspose2d layer
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        # Store HardTanh limits for the fused kernel
        self.hardtanh_min = float(hardtanh_min)
        self.hardtanh_max = float(hardtanh_max)

    def forward(self, x):
        # 1. ConvTranspose2d (native)
        x = self.conv_transpose(x)

        # 2-5. Fused MaxPool2d + HardTanh + Mean + Tanh (CUDA)
        x = fused_ops.fused_pool_hardtanh_mean_tanh_cuda(
            x,
            self.hardtanh_min,
            self.hardtanh_max,
        )
        return x
