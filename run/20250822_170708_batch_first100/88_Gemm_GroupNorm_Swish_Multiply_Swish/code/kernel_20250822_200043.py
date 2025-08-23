import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Inline CUDA/C++ source for the fused swish-multiply-swish operation
source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// Sigmoid function
__device__ float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Fused kernel: out = swish(x) * multiply_weight, then swish again
// swish(a) = a * sigmoid(a)
__global__ void fused_swish_mul_kernel(const float* __restrict__ x,
                                       const float* __restrict__ mul_w,
                                       float* __restrict__ out,
                                       int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        float s1 = sigmoidf(val);
        // first swish
        val = val * s1;
        // multiply by weight
        val = val * mul_w[idx];
        // second swish
        float s2 = sigmoidf(val);
        out[idx] = val * s2;
    }
}

torch::Tensor fused_swish_mul_cuda(torch::Tensor x, torch::Tensor mul_w) {
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;

    fused_swish_mul_kernel<<<grid_size, block_size>>>(x.data_ptr<float>(),
                                                      mul_w.data_ptr<float>(),
                                                      out.data_ptr<float>(),
                                                      size);

    return out;
}
"""

cpp_src = r"""
torch::Tensor fused_swish_mul_cuda(torch::Tensor x, torch::Tensor mul_w);
"""

# Build the fused kernel
fused_swish_mul = load_inline(
    name="fused_swish_mul",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["fused_swish_mul_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs:
    1. GEMM
    2. GroupNorm
    3. Fused swish -> multiply -> swish
    """
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))
        self.fused_swish_mul = fused_swish_mul

    def forward(self, x):
        x = self.gemm(x)
        x = self.group_norm(x)
        # fused swish(x), multiply by weight, swish again
        x = self.fused_swish_mul.fused_swish_mul_cuda(x, self.multiply_weight)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192
num_groups = 256
multiply_weight_shape = (out_features,)

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, num_groups, multiply_weight_shape]
