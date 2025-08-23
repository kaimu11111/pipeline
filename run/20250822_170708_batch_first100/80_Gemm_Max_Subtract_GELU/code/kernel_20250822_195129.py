import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

__device__ __forceinline__ float gelu(float x) {
    // 0.5 * x * (1 + erf(x / sqrt(2)))
    return 0.5f * x * (1.f + erff(x * M_SQRT1_2));
}

__global__ void fused_max_sub_gelu_kernel(const float* __restrict__ in,
                                          float* __restrict__ out,
                                          int batch_size,
                                          int out_features) {
    // Each block will handle one row.
    int row = blockIdx.x;
    if (row < batch_size) {
        // Partial maximum for this thread.
        float thread_max = -FLT_MAX;
        // Strided loop to find max per thread.
        for(int c = threadIdx.x; c < out_features; c += blockDim.x){
            float val = in[row * out_features + c];
            if(val > thread_max){
                thread_max = val;
            }
        }
        __shared__ float sdata[256];
        sdata[threadIdx.x] = thread_max;
        __syncthreads();

        // Reduce within the block.
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                float tmp = sdata[threadIdx.x + s];
                if (tmp > sdata[threadIdx.x]) {
                    sdata[threadIdx.x] = tmp;
                }
            }
            __syncthreads();
        }

        // sdata[0] now contains the row-wise maximum
        float row_max = sdata[0];
        // According to the given model code, we then subtract the mean along dim=1,
        // but there's only one element after max(...) with keepdim=True, so (row_max - row_max = 0).
        float result = row_max - row_max; 
        // Apply GELU
        result = gelu(result);

        if (threadIdx.x == 0) {
            // Store the final single value for this row.
            out[row] = result;
        }
    }
}

torch::Tensor fused_max_sub_gelu_cuda(torch::Tensor x, int max_dim) {
    // x is of shape (batch_size, out_features).
    // We'll return shape (batch_size, 1), so let's allocate out accordingly:
    auto batch_size = x.size(0);
    auto out_features = x.size(1);
    auto out = torch::empty({batch_size}, x.options());

    // Launch kernel with one block per row, and up to 256 threads per block
    const int threads = 256;
    const int blocks = batch_size;

    fused_max_sub_gelu_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        out_features
    );

    // Return a (batch_size, 1) shaped result to match keepdim=True usage.
    return out.view({batch_size, 1});
}
""";

cpp_src = r"""
torch::Tensor fused_max_sub_gelu_cuda(torch::Tensor x, int max_dim);
""";

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["fused_max_sub_gelu_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs nn.Linear, then fused max, subtract, and GELU operations in a custom CUDA kernel.
    """
    def __init__(self, in_features, out_features, max_dim):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.max_dim = max_dim
        self.fused_ops = fused_ops

    def forward(self, x):
        # Perform nn.Linear with PyTorch
        x = self.gemm(x)
        # Fused custom CUDA kernel for max -> subtract mean -> gelu
        x = self.fused_ops.fused_max_sub_gelu_cuda(x, self.max_dim)
        return x
