import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA source for fused GEMM + bias + scale
source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void gemm_scale_kernel(const float* __restrict__ input,
                                  const float* __restrict__ weight,
                                  const float* __restrict__ bias,
                                  const float* __restrict__ scale,
                                  float* __restrict__ output,
                                  int batch_size,
                                  int in_features,
                                  int out_features) {
    // 2D indexing: row -> batch index, col -> out_features index
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < batch_size && col < out_features) {
        float val = 0.0f;
        int weight_offset = col * in_features;
        int input_offset = row * in_features;
        for (int k = 0; k < in_features; ++k) {
            val += input[input_offset + k] * weight[weight_offset + k];
        }
        // Add bias, apply scale
        val += bias[col];
        val *= scale[col];
        // Write to output
        output[row * out_features + col] = val;
    }
}

torch::Tensor gemm_scale_cuda(torch::Tensor input,
                              torch::Tensor weight,
                              torch::Tensor bias,
                              torch::Tensor scale) {
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);

    auto output = torch::zeros({batch_size, out_features}, torch::CUDA(torch::kFloat));

    const int BLOCK_DIM = 16;
    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((out_features + BLOCK_DIM - 1) / BLOCK_DIM,
              (batch_size + BLOCK_DIM - 1) / BLOCK_DIM);

    gemm_scale_kernel<<<grid, block>>>(input.data_ptr<float>(),
                                       weight.data_ptr<float>(),
                                       bias.data_ptr<float>(),
                                       scale.data_ptr<float>(),
                                       output.data_ptr<float>(),
                                       batch_size,
                                       in_features,
                                       out_features);

    return output;
}
'''

cpp_src = r'''
torch::Tensor gemm_scale_cuda(torch::Tensor input,
                              torch::Tensor weight,
                              torch::Tensor bias,
                              torch::Tensor scale);
'''

gemm_scale = load_inline(
    name="gemm_scale",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=['gemm_scale_cuda'],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[]
)

class ModelNew(nn.Module):
    """
    Optimized model that fuses GEMM (weight/bias) and scale into a single custom CUDA kernel,
    then applies batch normalization.
    """
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        # Parameters to replace nn.Linear: weight, bias
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        # Scale parameter
        self.scale = nn.Parameter(torch.randn(scale_shape))
        # BatchNorm
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)

    def forward(self, x):
        # Fused GEMM + bias + scale
        out = gemm_scale.gemm_scale_cuda(
            x.contiguous(),
            self.weight.contiguous(),
            self.bias.contiguous(),
            self.scale.contiguous()
        )
        # Batch normalization
        out = self.bn(out)
        return out
