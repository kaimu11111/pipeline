import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void compute_statistics_kernel(
    const float* __restrict__ input,
    float* sums,
    float* sums_sq,
    int N, int C, int H, int W
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N*C*H*W;
    if (idx < total){
        float val = input[idx];
        int c_ = (idx / (H*W)) % C;
        int n_ =  idx / (C*H*W);
        int offset = n_ * C + c_;
        atomicAdd(&sums[offset], val);
        atomicAdd(&sums_sq[offset], val * val);
    }
}

__global__ void apply_instance_norm_divide_kernel(
    const float* __restrict__ input,
    float* output,
    const float* sums,
    const float* sums_sq,
    int N, int C, int H, int W,
    float eps,
    float divide_by
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N*C*H*W;
    if (idx < total){
        int c_ = (idx / (H*W)) % C;
        int n_ =  idx / (C*H*W);
        int offset = n_ * C + c_;
        int spatial_size = H * W;
        float mean = sums[offset] / spatial_size;
        float var  = sums_sq[offset] / spatial_size - mean * mean;
        float val = input[idx];
        float normed = (val - mean) / sqrtf(var + eps);
        output[idx] = normed / divide_by;
    }
}

torch::Tensor fused_instance_norm_divide_cuda(torch::Tensor input, float divide_by){
    TORCH_CHECK(input.dim() == 4, "Input must be a 4D tensor");
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    auto sums = torch::zeros({N*C}, input.options());
    auto sums_sq = torch::zeros({N*C}, input.options());
    auto output = torch::zeros_like(input);

    int total = N*C*H*W;
    const int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;

    compute_statistics_kernel<<<gridSize, blockSize>>>(
        input.data_ptr<float>(),
        sums.data_ptr<float>(),
        sums_sq.data_ptr<float>(),
        N, C, H, W
    );
    apply_instance_norm_divide_kernel<<<gridSize, blockSize>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        sums.data_ptr<float>(),
        sums_sq.data_ptr<float>(),
        N, C, H, W,
        1e-5f,
        divide_by
    );

    return output;
}
"""

cpp_src = r"""
torch::Tensor fused_instance_norm_divide_cuda(torch::Tensor input, float divide_by);
"""

fused_instance_norm_divide = load_inline(
    name="fused_instance_norm_divide",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["fused_instance_norm_divide_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divide_by = divide_by

    def forward(self, x):
        x = self.conv(x)
        x = fused_instance_norm_divide.fused_instance_norm_divide_cuda(x, self.divide_by)
        return x
