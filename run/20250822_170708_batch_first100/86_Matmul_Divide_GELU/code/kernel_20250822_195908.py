import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ float approximate_gelu(float x) {
    // Approximate GELU
    const float kAlpha = 0.7978845608f; 
    const float kBeta = 0.044715f; 
    float x3 = x * x * x; 
    return 0.5f * x * (1.f + tanhf(kAlpha * (x + kBeta * x3)));
}

__global__ void fused_gemm_div_gelu_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ out,
    const int batch_size,
    const int input_size,
    const int output_size,
    const float divisor
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < batch_size && j < output_size) {
        float val = 0.0f;
        // Compute matmul: x[i, :] * w[j, :]
        // (Remember w is stored as (out_features, in_features))
        for (int k = 0; k < input_size; k++) {
            val += x[i * input_size + k] * w[j * input_size + k];
        }
        // Add bias, divide, apply GELU
        val += b[j];
        val /= divisor;
        val = approximate_gelu(val);
        out[i * output_size + j] = val;
    }
}

torch::Tensor fused_gemm_div_gelu_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b,
    float divisor
) {
    auto batch_size = x.size(0);
    auto input_size = x.size(1);
    auto output_size = w.size(0);

    auto out = torch::zeros({batch_size, output_size}, x.options());

    dim3 block(16, 16);
    dim3 grid((batch_size + block.x - 1) / block.x,
              (output_size + block.y - 1) / block.y);

    fused_gemm_div_gelu_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        input_size,
        output_size,
        divisor
    );

    return out;
}
"""

cpp_src = r"""
torch::Tensor fused_gemm_div_gelu_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b,
    float divisor
);
"""

fused_op = load_inline(
    name="fused_gemm_div_gelu",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["fused_gemm_div_gelu_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model that fuses matmul, division by scalar, and GELU into a single CUDA kernel.
    """
    def __init__(self, input_size, output_size, divisor):
        super(ModelNew, self).__init__()
        # Match nn.Linear parameter shapes
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.bias = nn.Parameter(torch.empty(output_size))
        self.divisor = divisor
        self.reset_parameters()

    def reset_parameters(self):
        # Mimic default nn.Linear initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Fused GEMM + division + GELU
        return fused_op.fused_gemm_div_gelu_cuda(x, self.weight, self.bias, self.divisor)

batch_size = 1024
input_size = 8192
output_size = 8192
divisor = 10.0

def get_inputs():
    return [torch.rand(batch_size, input_size).cuda()]

def get_init_inputs():
    return [input_size, output_size, divisor]
