import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ----------------------------------------------------------------------
# CUDA kernel: fused 2×2 max-pool → hardtanh → spatial mean → tanh
# ----------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

__global__ void fused_kernel(const float* __restrict__ input,
                             float* __restrict__ output,
                             int N, int C, int H, int W,
                             int pool_k, int pool_stride,
                             float ht_min, float ht_max,
                             int pooled_H, int pooled_W) {
    /* One (n,c) pair per block */
    const int nc = blockIdx.x;
    const int n  = nc / C;
    const int c  = nc % C;

    const int tid         = threadIdx.x;
    const int num_threads = blockDim.x;
    const int windows     = pooled_H * pooled_W;   // number of pooling windows

    extern __shared__ float sdata[];
    float acc = 0.f;

    /* Iterate over pooling windows assigned to this thread */
    for (int idx = tid; idx < windows; idx += num_threads) {
        const int ph = idx / pooled_W;
        const int pw = idx % pooled_W;

        float max_val = -FLT_MAX;
        /* 2-D pooling window */
        for (int kh = 0; kh < pool_k; ++kh) {
            for (int kw = 0; kw < pool_k; ++kw) {
                int h = ph * pool_stride + kh;
                int w = pw * pool_stride + kw;
                int offset = (((n * C + c) * H + h) * W) + w;
                float v = input[offset];
                if (v > max_val) max_val = v;
            }
        }
        /* hardtanh */
        max_val = max(ht_min, min(ht_max, max_val));
        acc += max_val;
    }

    /* Block reduction in shared memory */
    sdata[tid] = acc;
    __syncthreads();
    for (int stride = num_threads >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    /* Write result: mean → tanh */
    if (tid == 0) {
        float mean = sdata[0] / (float)windows;
        output[n * C + c] = tanhf(mean);
    }
}

torch::Tensor fused_pool_hardtanh_mean_tanh(torch::Tensor input,
                                            int64_t  pool_k,
                                            int64_t  pool_stride,
                                            double   ht_min,
                                            double   ht_max) {
    TORCH_CHECK(input.is_cuda(), "Input must reside on CUDA device");
    input = input.contiguous();

    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);

    const int pooled_H = (H - pool_k) / pool_stride + 1;
    const int pooled_W = (W - pool_k) / pool_stride + 1;

    auto output = torch::empty({N, C}, input.options());

    const int blocks   = N * C;
    const int threads  = 256;
    const size_t shmem = threads * sizeof(float);

    fused_kernel<<<blocks, threads, shmem>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        static_cast<int>(pool_k),
        static_cast<int>(pool_stride),
        static_cast<float>(ht_min),
        static_cast<float>(ht_max),
        pooled_H, pooled_W
    );

    return output.view({N, C, 1, 1});
}
"""

cpp_source = """
torch::Tensor fused_pool_hardtanh_mean_tanh(torch::Tensor input,
                                            int64_t  pool_k,
                                            int64_t  pool_stride,
                                            double   ht_min,
                                            double   ht_max);
"""

fused_ops = load_inline(
    name="fused_pool_hardtanh_mean_tanh",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["fused_pool_hardtanh_mean_tanh"],
    verbose=False,
)

# ----------------------------------------------------------------------
# Optimised model
# ----------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Optimised version of the reference model.
    The transposed convolution is kept as-is, while max-pool, hardtanh,
    spatial mean and tanh are fused into a single CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding
        )
        self.pool_k   = maxpool_kernel_size
        self.pool_s   = maxpool_stride
        self.ht_min   = hardtanh_min
        self.ht_max   = hardtanh_max
        self.fused_op = fused_ops

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_op.fused_pool_hardtanh_mean_tanh(
            x, self.pool_k, self.pool_s, self.ht_min, self.ht_max
        )
        return x

# ----------------------------------------------------------------------
# Helper functions required by the harness
# ----------------------------------------------------------------------
def get_inputs():
    batch_size   = 64
    in_channels  = 32
    height = width = 128
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    in_channels          = 32
    out_channels         = 32
    kernel_size          = 3
    stride               = 1
    padding              = 1
    maxpool_kernel_size  = 2
    maxpool_stride       = 2
    hardtanh_min         = -1
    hardtanh_max         = 1
    return [in_channels, out_channels, kernel_size, stride, padding,
            maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max]
