import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Inline CUDA source for fused linear + min + subtract
source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_linear_min_sub_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ out,
    const int batch_size,
    const int in_features,
    const int out_features,
    const float min_constant
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < batch_size && col < out_features) {
        float val = 0.0f;
        // Compute matmul
        for (int k = 0; k < in_features; k++) {
            val += x[row * in_features + k] * w[col * in_features + k];
        }
        // Add bias
        val += b[col];
        // Apply min with the constant
        val = fminf(val, min_constant);
        // Subtract the same constant
        val -= min_constant;
        // Write result
        out[row * out_features + col] = val;
    }
}

torch::Tensor fused_linear_min_sub_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b,
    float min_constant
) {
    // x: [batch_size, in_features]
    // w: [out_features, in_features]
    // b: [out_features]
    // out: [batch_size, out_features]

    // Check device
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(w.is_cuda(), "w must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");

    int batch_size = x.size(0);
    int in_features = x.size(1);
    int out_features = w.size(0);

    auto out = torch::zeros({batch_size, out_features}, x.options());

    const dim3 block(16, 16);
    const dim3 grid((batch_size + block.x - 1) / block.x,
                    (out_features + block.y - 1) / block.y);

    fused_linear_min_sub_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        min_constant
    );
    return out;
}
"""

cpp_src = r"""
torch::Tensor fused_linear_min_sub_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b,
    float min_constant
);
"""

# Build/load the fused operator
fused_op = load_inline(
    name="fused_linear_min_sub",
    cpp_sources=cpp_src,
    cuda_sources=source,
    extra_cflags=[],
    extra_ldflags=[],
    functions=["fused_linear_min_sub_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model that uses a fused CUDA kernel for:
    matmul + bias + min + subtract.
    """
    def __init__(self, in_features, out_features, constant):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        # Store same constant as a learnable parameter for consistency
        self.constant = nn.Parameter(torch.tensor(constant, device='cuda', dtype=torch.float32))

    def forward(self, x):
        return fused_op.fused_linear_min_sub_cuda(
            x, self.linear.weight, self.linear.bias, float(self.constant.item())
        )
