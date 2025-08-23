import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# This CUDA source implements a fused operator that performs:
# 1) MaxPool1D with kernel_size=2 on each row of x (shape: [batch_size, features])
# 2) Sums the pooled values along the feature dimension
# 3) Multiplies by a scale factor
# The result is a 1D tensor of shape (batch_size).
source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

__device__ __forceinline__ void warpReduceSum(volatile float* sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void fused_maxpool_sum_scale_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    const int batch_size,
    const int features,
    const int kernel_size,
    const float scale_factor
) {
    // Each block handles one batch element
    const int b = blockIdx.x;
    // Shared memory for block reduction
    extern __shared__ float sdata[];
    
    float partial_sum = 0.0f;
    // We have (features / kernel_size) "pooled" values per batch element
    int pooled_size = features / kernel_size;
    
    for (int idx = threadIdx.x; idx < pooled_size; idx += blockDim.x) {
        int start = b * features + (idx * kernel_size);
        float mval = x[start];
        // Max-pool over kernel_size elements
        #pragma unroll
        for (int k = 1; k < kernel_size; k++) {
            float val = x[start + k];
            mval = (val > mval) ? val : mval;
        }
        partial_sum += mval;
    }

    sdata[threadIdx.x] = partial_sum;
    __syncthreads();

    // Parallel reduction within the block
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Warp-level reduction
    if (threadIdx.x < 32) {
        warpReduceSum(sdata, threadIdx.x);
    }

    // Write out to global memory
    if (threadIdx.x == 0) {
        out[b] = sdata[0] * scale_factor;
    }
}

torch::Tensor fused_maxpool_sum_scale(
    torch::Tensor x,
    int kernel_size,
    float scale_factor
) {
    // x shape: [batch_size, features]
    TORCH_CHECK(x.dim() == 2, "Input must be 2D");
    auto batch_size = x.size(0);
    auto features = x.size(1);

    auto opts = x.options().dtype(x.dtype());
    auto out = torch::empty({batch_size}, opts);

    // We launch one block per batch element
    const int threads = 256;
    const int shared_mem = threads * sizeof(float);

    fused_maxpool_sum_scale_kernel<<<batch_size, threads, shared_mem>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        features,
        kernel_size,
        scale_factor
    );
    return out;
}
"""

cpp_src = r"""
torch::Tensor fused_maxpool_sum_scale(
    torch::Tensor x,
    int kernel_size,
    float scale_factor
);
"""

# Load the inline extension containing the fused operator
fused_ops = load_inline(
    name="fused_maxpool_sum_scale_extension",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["fused_maxpool_sum_scale"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model: uses a standard Linear for matmul, then a fused custom CUDA kernel
    to perform MaxPool1D (kernel=2), sum over features, and scale.
    """
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.matmul(x)
        x = fused_ops.fused_maxpool_sum_scale(x, self.kernel_size, self.scale_factor)
        return x

batch_size = 128
in_features = 32768
out_features = 32768
kernel_size = 2
scale_factor = 0.5

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, kernel_size, scale_factor]
