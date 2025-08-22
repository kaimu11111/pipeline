import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_gemm_leaky_relu_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int batch_size,
    int in_features,
    int out_features,
    float multiplier,
    float negative_slope)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size && col < out_features) {
        float val = 0.0f;
        for (int i = 0; i < in_features; i++) {
            val += x[row * in_features + i] * w[col * in_features + i];
        }
        val += bias[col];
        val *= multiplier;
        if (val < 0.0f) {
            val *= negative_slope;
        }
        out[row * out_features + col] = val;
    }
}

torch::Tensor fused_gemm_leaky_relu_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor bias,
    float multiplier,
    float negative_slope)
{
    auto batch_size = x.size(0);
    auto in_features = x.size(1);
    auto out_features = w.size(0);

    // create output
    auto out = torch::empty({batch_size, out_features}, x.options());

    // set up block and grid
    dim3 block(16, 16);
    dim3 grid((out_features + block.x - 1) / block.x,
              (batch_size + block.y - 1) / block.y);

    // launch kernel
    fused_gemm_leaky_relu_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        multiplier,
        negative_slope
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "fused_gemm_leaky_relu_cuda",
        &fused_gemm_leaky_relu_cuda,
        "Fused Gemm + LeakyReLU kernel"
    );
}
"""

cpp_src = r"""
torch::Tensor fused_gemm_leaky_relu_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor bias,
    float multiplier,
    float negative_slope
);
"""

fused_gemm_leaky_relu = load_inline(
    name="fused_gemm_leaky_relu",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["fused_gemm_leaky_relu_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    """
    Optimized model with fused cuda kernel for Gemm + multiply + LeakyReLU
    """
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(ModelNew, self).__init__()
        # Emulate nn.Linear initialization
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

        self.multiplier = multiplier
        self.negative_slope = negative_slope

    def forward(self, x):
        return fused_gemm_leaky_relu.fused_gemm_leaky_relu_cuda(
            x, self.weight, self.bias, self.multiplier, self.negative_slope
        )

batch_size = 1024
in_features = 8192
out_features = 8192
multiplier = 2.0
negative_slope = 0.1

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, multiplier, negative_slope]
