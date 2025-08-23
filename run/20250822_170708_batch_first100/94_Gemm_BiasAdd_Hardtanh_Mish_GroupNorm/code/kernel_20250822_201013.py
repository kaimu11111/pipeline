import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Inline CUDA code for fused bias-add, Hardtanh, Mish
fused_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// Fused kernel (bias-add -> Hardtanh -> Mish)
__global__ void fused_kernel(
    float* data,
    const float* bias,
    const int N,
    const int features,
    const float min_val,
    const float max_val)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int c = idx % features;
        float val = data[idx] + bias[c];

        // Hardtanh
        if (val < min_val) val = min_val;
        else if (val > max_val) val = max_val;

        // Mish = x * tanh(softplus(x))
        float sp = logf(1.0f + expf(val));
        float t = tanhf(sp);
        val = val * t;
        data[idx] = val;
    }
}

torch::Tensor fused_activation_cuda(
    torch::Tensor input,
    torch::Tensor bias,
    float min_val,
    float max_val)
{
    auto out = input.clone();
    int N = out.numel();
    int features = bias.size(0);

    const int block_size = 256;
    const int num_blocks = (N + block_size - 1) / block_size;

    fused_kernel<<<num_blocks, block_size>>>(
        out.data_ptr<float>(),
        bias.data_ptr<float>(),
        N,
        features,
        min_val,
        max_val
    );
    return out;
}
""".strip()

fused_cpp_src = r"""
torch::Tensor fused_activation_cuda(
    torch::Tensor input,
    torch::Tensor bias,
    float min_val,
    float max_val);
"""

# Build extension
fused_activation = load_inline(
    name="fused_activation",
    cpp_sources=fused_cpp_src,
    cuda_sources=fused_source,
    functions=["fused_activation_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    """
    A model that performs a GEMM, then a fused bias-add/Hardtanh/Mish step,
    followed by a standard PyTorch GroupNorm.
    """
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=out_features)

    def forward(self, x):
        x = self.gemm(x)
        # Fused bias-add -> Hardtanh -> Mish
        x = fused_activation.fused_activation_cuda(x, self.bias, -1.0, 1.0)
        # GroupNorm
        x = self.groupnorm(x)
        return x
