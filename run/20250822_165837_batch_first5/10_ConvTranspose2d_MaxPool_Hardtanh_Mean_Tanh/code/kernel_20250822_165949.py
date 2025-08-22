import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_src = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>
#include <cmath>

__global__ void maxpool_hardtanh_kernel(
    const float* __restrict__ input, 
    float* __restrict__ output,
    int B, int C, int inH, int inW, 
    int kernel_size, int stride, 
    float min_val, float max_val)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int outH = inH / stride;
    int outW = inW / stride;
    int total = B * C * outH * outW;
    if (idx >= total) return;

    int ow = idx % outW;
    int oh = (idx / outW) % outH;
    int c = (idx / outW / outH) % C;
    int b = idx / (outW * outH * C);

    int h_start = oh * stride;
    int w_start = ow * stride;

    float pool_val = -FLT_MAX;
    for(int kh = 0; kh < kernel_size; kh++) {
        for(int kw = 0; kw < kernel_size; kw++) {
            int h_in = h_start + kh;
            int w_in = w_start + kw;
            if (h_in < inH && w_in < inW) {
                float val = input[((b*C + c)*inH + h_in)*inW + w_in];
                if (val > pool_val) {
                    pool_val = val;
                }
            }
        }
    }
    // clamp
    if (pool_val < min_val) pool_val = min_val;
    if (pool_val > max_val) pool_val = max_val;

    output[idx] = pool_val;
}

__global__ void mean_tanh_kernel(
    const float* __restrict__ input, 
    float* __restrict__ output,
    int B, int C, int H, int W)
{
    int bc_idx = blockIdx.x;
    if (bc_idx >= B*C) return;

    int b = bc_idx / C;
    int c = bc_idx % C;

    float sum_val = 0.f;
    for(int i = threadIdx.x; i < H*W; i += blockDim.x) {
        int h = i / W;
        int w = i % W;
        sum_val += input[((b*C + c)*H + h)*W + w];
    }

    __shared__ float s[256];
    s[threadIdx.x] = sum_val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s[threadIdx.x] += s[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float meanval = s[0] / (H * W);
        output[bc_idx] = tanhf(meanval);
    }
}

torch::Tensor maxpool_hardtanh_cuda(
    torch::Tensor input, 
    int kernel_size, 
    int stride,
    float min_val, 
    float max_val)
{
    auto B = input.size(0);
    auto C = input.size(1);
    auto H = input.size(2);
    auto W = input.size(3);

    int outH = H / stride;
    int outW = W / stride;

    auto out = torch::zeros({B, C, outH, outW}, input.options());
    int block_size = 256;
    int grid_size = (B*C*outH*outW + block_size - 1) / block_size;

    maxpool_hardtanh_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(), 
        out.data_ptr<float>(),
        B, C, H, W, 
        kernel_size, stride, 
        min_val, max_val
    );

    return out;
}

torch::Tensor mean_tanh_cuda(torch::Tensor input)
{
    auto B = input.size(0);
    auto C = input.size(1);
    auto H = input.size(2);
    auto W = input.size(3);

    auto out = torch::zeros({B, C, 1, 1}, input.options());
    dim3 grid(B*C);
    dim3 block(256);

    mean_tanh_kernel<<<grid, block>>>(
        input.data_ptr<float>(), 
        out.data_ptr<float>(),
        B, C, H, W
    );

    return out;
}
'''

cpp_src = r'''
torch::Tensor maxpool_hardtanh_cuda(
    torch::Tensor input, 
    int kernel_size, 
    int stride,
    float min_val, 
    float max_val);

torch::Tensor mean_tanh_cuda(torch::Tensor input);
'''

model_new_extension = load_inline(
    name="model_new_extension",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["maxpool_hardtanh_cuda", "mean_tanh_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a transposed convolution, 
    followed by a custom fused maxpool+hardtanh kernel, 
    and then a custom mean+tanh kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, 
                 maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding
        )
        self.maxpool_kernel_size = maxpool_kernel_size
        self.maxpool_stride = maxpool_stride
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

    def forward(self, x):
        x = self.conv_transpose(x)
        x = model_new_extension.maxpool_hardtanh_cuda(
            x, self.maxpool_kernel_size, self.maxpool_stride, 
            self.hardtanh_min, self.hardtanh_max
        )
        x = model_new_extension.mean_tanh_cuda(x)
        return x
