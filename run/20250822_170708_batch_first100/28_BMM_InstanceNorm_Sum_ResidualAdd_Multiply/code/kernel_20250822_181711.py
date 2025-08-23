import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA source code
source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// Naive instance norm kernel: for each batch element, compute mean and var over features
__global__ void instance_norm_kernel(const float* __restrict__ input, 
                                     float* __restrict__ output,
                                     int batch_size,
                                     int feature_size,
                                     float eps) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch_size) {
        // Compute mean
        double mean = 0.0;
        for (int i = 0; i < feature_size; i++) {
            mean += input[b * feature_size + i];
        }
        mean /= feature_size;

        // Compute variance
        double var = 0.0;
        for (int i = 0; i < feature_size; i++) {
            double diff = input[b * feature_size + i] - mean;
            var += diff * diff;
        }
        var /= feature_size;
        float inv_std = 1.0f / sqrtf(var + eps);

        // Normalize
        for (int i = 0; i < feature_size; i++) {
            output[b * feature_size + i] = (input[b * feature_size + i] - mean) * inv_std;
        }
    }
}

// Fused add+mul kernel: out = (x + y) * y
__global__ void fused_add_mul_kernel(const float* __restrict__ x,
                                     const float* __restrict__ y,
                                     float* __restrict__ out,
                                     int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx] + y[idx];
        out[idx] = val * y[idx];
    }
}

torch::Tensor instance_norm_cuda(torch::Tensor x, double eps) {
    // x shape: (batch_size, feature_size)
    auto sizes = x.sizes();
    int batch_size = sizes[0];
    int feature_size = sizes[1];
    
    auto out = torch::empty_like(x);

    // Launch one thread per batch element, naive approach
    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;

    instance_norm_kernel<<<blocks, threads>>>(x.data_ptr<float>(),
                                             out.data_ptr<float>(),
                                             batch_size,
                                             feature_size,
                                             (float)eps);
    return out;
}

torch::Tensor fused_add_mul_cuda(torch::Tensor x, torch::Tensor y) {
    // x, y shape: (batch_size, feature_size)
    auto size = x.numel();
    auto out = torch::empty_like(x);

    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    fused_add_mul_kernel<<<blocks, threads>>>(x.data_ptr<float>(),
                                             y.data_ptr<float>(),
                                             out.data_ptr<float>(),
                                             size);
    return out;
}
""";

# C++ function signatures for Python bindings
cpp_src = r"""
torch::Tensor instance_norm_cuda(torch::Tensor x, double eps);
torch::Tensor fused_add_mul_cuda(torch::Tensor x, torch::Tensor y);
"""

# Build the extension
cuda_ops = load_inline(
    name="custom_cuda_ops",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["instance_norm_cuda", "fused_add_mul_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA operators for instance normalization and fused addition/multiplication.
    """
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        # We'll still use PyTorch's Linear for the bmm operation
        self.bmm = nn.Linear(in_features, out_features)
        self.eps = eps
        # We'll skip the momentum tracking logic and affine params,
        # using a custom kernel instead of nn.InstanceNorm2d
        # Momentum is not used in this naive custom kernel.

    def forward(self, x, y):
        # Perform the linear transformation
        x = self.bmm(x)
        # Apply custom instance norm
        x = cuda_ops.instance_norm_cuda(x, self.eps)
        # Fused elementwise: (x + y) * y
        x = cuda_ops.fused_add_mul_cuda(x, y)
        return x

batch_size = 1024
in_features = 8192
out_features = 8192

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda(), torch.rand(batch_size, out_features).cuda()]

def get_init_inputs():
    return [in_features, out_features]
