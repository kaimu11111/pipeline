import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA code
source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void naive_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_channels,
    int in_height, int in_width,
    int out_channels, int kernel_size,
    int out_height, int out_width)
{
    // Block and thread indexing
    // blockIdx.x ranges over out_height * out_width
    // blockIdx.y ranges over out_channels
    // blockIdx.z ranges over batch_size
    int n  = blockIdx.z;           // batch index
    int oc = blockIdx.y;           // out channel index
    int idx = blockIdx.x;          // combined out_height/out_width index
    int oh = idx / out_width;      // out height index
    int ow = idx % out_width;      // out width index

    float val = bias[oc];
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int ih = oh + kh;
                int iw = ow + kw;
                float inp = input[n * (in_channels * in_height * in_width)
                                  + ic * (in_height * in_width)
                                  + ih * in_width
                                  + iw];
                float w = weight[oc * (in_channels * kernel_size * kernel_size)
                                 + ic * (kernel_size * kernel_size)
                                 + kh * kernel_size
                                 + kw];
                val += inp * w;
            }
        }
    }
    output[n * (out_channels * out_height * out_width)
           + oc * (out_height * out_width)
           + oh * out_width
           + ow] = val;
}

__device__ __forceinline__ float fast_gelu(float x) {
    const float kAlpha = 0.7978845608f;   // sqrt(2.0/M_PI)
    const float kBeta = 0.044715f;
    return 0.5f * x * (1.0f + tanhf(kAlpha * (x + kBeta * x * x * x)));
}

__global__ void gelu_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fast_gelu(data[idx]);
    }
}

__global__ void global_avg_pool_kernel(
    const float* __restrict__ input,
    float* __restrict__ out,
    int batch_size, int out_channels,
    int out_height, int out_width)
{
    // Each block handles one batch element,
    // each thread within that block handles one out_channel.
    int n = blockIdx.x;    // batch index
    int oc = threadIdx.x;  // channel index
    if (n < batch_size && oc < out_channels) {
        float sum_val = 0.0f;
        int area = out_height * out_width;
        for (int oh = 0; oh < out_height; oh++) {
            for (int ow = 0; ow < out_width; ow++) {
                sum_val += input[n * (out_channels * out_height * out_width)
                                 + oc * (out_height * out_width)
                                 + oh * out_width
                                 + ow];
            }
        }
        out[n * out_channels + oc] = sum_val / (float)area;
    }
}

// Wrapper for naive_conv2d
torch::Tensor naive_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias)
{
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);

    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);

    int out_height = in_height - kernel_size + 1;
    int out_width = in_width - kernel_size + 1;

    auto options = torch::TensorOptions()
        .dtype(input.dtype())
        .device(input.device());
    auto out = torch::zeros({batch_size, out_channels, out_height, out_width}, options);

    // Each block handles one output pixel (oh,ow) and one out_channel, for each batch
    dim3 blocks(out_height * out_width, out_channels, batch_size);
    dim3 threads(1);

    naive_conv2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size, in_channels, in_height, in_width,
        out_channels, kernel_size,
        out_height, out_width
    );

    return out;
}

// Wrapper for GELU
torch::Tensor gelu_cuda(torch::Tensor data) {
    int size = data.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    gelu_kernel<<<blocks, threads>>>(data.data_ptr<float>(), size);
    return data;
}

// Wrapper for global average pooling (to 1x1)
torch::Tensor global_avg_pool_cuda(torch::Tensor data) {
    int batch_size = data.size(0);
    int out_channels = data.size(1);
    int out_height = data.size(2);
    int out_width = data.size(3);

    auto options = torch::TensorOptions()
        .dtype(data.dtype())
        .device(data.device());
    auto out = torch::zeros({batch_size, out_channels}, options);

    global_avg_pool_kernel<<<batch_size, out_channels>>>(
        data.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size, out_channels, out_height, out_width
    );

    return out;
}
""".strip()

cpp_src = r"""
torch::Tensor naive_conv2d_cuda(torch::Tensor input,
                                torch::Tensor weight,
                                torch::Tensor bias);
torch::Tensor gelu_cuda(torch::Tensor data);
torch::Tensor global_avg_pool_cuda(torch::Tensor data);
"""

# Compile the inline CUDA code
my_custom_ops = load_inline(
    name="my_custom_ops",
    cpp_sources=cpp_src,
    cuda_sources=source,
    extra_cflags=["-O3"],
    extra_ldflags=[],
    functions=[
        "naive_conv2d_cuda",
        "gelu_cuda",
        "global_avg_pool_cuda"
    ],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model that manually performs a naive convolution, applies GELU, 
    and performs global average pooling with custom CUDA ops.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        # Create parameters to match the original Conv2D
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        # 1) Naive convolution
        out = my_custom_ops.naive_conv2d_cuda(x, self.weight, self.bias)
        # 2) GELU
        out = my_custom_ops.gelu_cuda(out)
        # 3) Global avg pool
        out = my_custom_ops.global_avg_pool_cuda(out)
        return out

batch_size = 128
in_channels = 8
out_channels = 64
height, width = 256, 256
kernel_size = 3

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device="cuda")]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
