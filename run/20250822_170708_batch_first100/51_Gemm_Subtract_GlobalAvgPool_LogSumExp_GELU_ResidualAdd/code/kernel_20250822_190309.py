import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Inline CUDA/C++ source with multiple custom operators
source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// GELU function
__device__ __forceinline__ float gelu_func(float x) {
    // exact GELU
    return 0.5f * x * (1.0f + erff(x / 1.41421356237f));
}

// Subtract operator
__global__ void subtract_kernel(const float* __restrict__ inp, const float* __restrict__ sub, float* __restrict__ out, int batch, int features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * features;
    if (idx < total) {
        int f = idx % features;
        out[idx] = inp[idx] - sub[f];
    }
}

// Global average pool operator
__global__ void global_avg_pool_kernel(const float* __restrict__ inp, float* __restrict__ out, int batch, int features) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch) {
        float sum_val = 0.0f;
        for (int f = 0; f < features; f++) {
            sum_val += inp[i * features + f];
        }
        out[i] = sum_val / features;
    }
}

// LogSumExp operator
__global__ void logsumexp_kernel(const float* __restrict__ inp, float* __restrict__ out, int batch, int features) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch) {
        // find max
        float max_val = inp[i * features];
        for (int f = 1; f < features; f++) {
            float temp = inp[i * features + f];
            if (temp > max_val) max_val = temp;
        }
        // sum of exp
        float sum_exp = 0.0f;
        for (int f = 0; f < features; f++) {
            sum_exp += expf(inp[i * features + f] - max_val);
        }
        out[i] = max_val + logf(sum_exp);
    }
}

// GELU operator
__global__ void gelu_kernel(const float* __restrict__ inp, float* __restrict__ out, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        float val = inp[idx];
        out[idx] = gelu_func(val);
    }
}

// PyTorch binding functions
torch::Tensor custom_subtract_cuda(torch::Tensor inp, torch::Tensor sub){
    int batch = inp.size(0);
    int features = inp.size(1);
    auto out = torch::empty_like(inp);
    int total = batch * features;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    subtract_kernel<<<grid_size, block_size>>>(inp.data_ptr<float>(), sub.data_ptr<float>(), out.data_ptr<float>(), batch, features);
    return out;
}

torch::Tensor global_avg_pool_cuda(torch::Tensor inp){
    int batch = inp.size(0);
    int features = inp.size(1);
    auto out = torch::zeros({batch}, inp.options());
    int block_size = 256;
    int grid_size = (batch + block_size - 1) / block_size;
    global_avg_pool_kernel<<<grid_size, block_size>>>(inp.data_ptr<float>(), out.data_ptr<float>(), batch, features);
    // reshape to [batch,1]
    return out.view({batch, 1});
}

torch::Tensor logsumexp_cuda(torch::Tensor inp){
    int batch = inp.size(0);
    int features = inp.size(1);
    auto out = torch::zeros({batch}, inp.options());
    int block_size = 256;
    int grid_size = (batch + block_size - 1) / block_size;
    logsumexp_kernel<<<grid_size, block_size>>>(inp.data_ptr<float>(), out.data_ptr<float>(), batch, features);
    // reshape to [batch,1]
    return out.view({batch, 1});
}

torch::Tensor gelu_cuda(torch::Tensor inp){
    int total = inp.numel();
    auto out = torch::empty_like(inp);
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    gelu_kernel<<<grid_size, block_size>>>(inp.data_ptr<float>(), out.data_ptr<float>(), total);
    return out;
}
"""

# Corresponding function declarations
cpp_src = r"""
torch::Tensor custom_subtract_cuda(torch::Tensor inp, torch::Tensor sub);
torch::Tensor global_avg_pool_cuda(torch::Tensor inp);
torch::Tensor logsumexp_cuda(torch::Tensor inp);
torch::Tensor gelu_cuda(torch::Tensor inp);
"""

# Build the inline extension
custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["custom_subtract_cuda",
               "global_avg_pool_cuda",
               "logsumexp_cuda",
               "gelu_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=["-O3"],
)

class ModelNew(nn.Module):
    """
    Optimized Model that leverages custom CUDA kernels for subtract,
    global average pool, logsumexp, and GELU, while keeping PyTorch's
    nn.Linear for the GEMM.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        original_x = x.clone().detach()
        # Gemm (nn.Linear)
        x = self.gemm(x)
        # Custom Subtract
        x = custom_ops.custom_subtract_cuda(x, self.subtract)
        # Custom GlobalAvgPool
        x = custom_ops.global_avg_pool_cuda(x)
        # Custom LogSumExp
        x = custom_ops.logsumexp_cuda(x)
        # Custom GELU
        x = custom_ops.gelu_cuda(x)
        # ResidualAdd (with broadcasting)
        x = x + original_x
        return x

# Keep these functions consistent with the original interface
batch_size = 2048
in_features = 8192
out_features = 8192

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features]
