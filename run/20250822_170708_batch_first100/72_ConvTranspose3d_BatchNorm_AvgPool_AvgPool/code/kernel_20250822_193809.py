import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 3D average pooling
source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void avg_pool_3d_kernel(const float* input, float* output,
                                   int N, int C, int D_in, int H_in, int W_in,
                                   int D_out, int H_out, int W_out,
                                   int kernel_d, int kernel_h, int kernel_w,
                                   int stride_d, int stride_h, int stride_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out_size = N * C * D_out * H_out * W_out;
    if (idx >= total_out_size) return;

    int w_out = idx % W_out;
    int temp = idx / W_out;
    int h_out = temp % H_out;
    temp = temp / H_out;
    int d_out = temp % D_out;
    temp = temp / D_out;
    int c = temp % C;
    int n = temp / C;

    int d_start = d_out * stride_d;
    int h_start = h_out * stride_h;
    int w_start = w_out * stride_w;

    float sum_val = 0.0;
    int count = 0;
    for(int kd = 0; kd < kernel_d; kd++){
        for(int kh = 0; kh < kernel_h; kh++){
            for(int kw = 0; kw < kernel_w; kw++){
                int d_in_idx = d_start + kd;
                int h_in_idx = h_start + kh;
                int w_in_idx = w_start + kw;
                if(d_in_idx < D_in && h_in_idx < H_in && w_in_idx < W_in){
                    int in_idx = (((n * C + c) * D_in + d_in_idx)
                                  * H_in + h_in_idx) * W_in + w_in_idx;
                    sum_val += input[in_idx];
                    count++;
                }
            }
        }
    }
    output[idx] = sum_val / count;
}

torch::Tensor average_pool3d_cuda(torch::Tensor input, int kernel_d, int kernel_h, int kernel_w){
    int stride_d = kernel_d;
    int stride_h = kernel_h;
    int stride_w = kernel_w;

    auto sizes = input.sizes(); // [N, C, D, H, W]
    int N = sizes[0];
    int C = sizes[1];
    int D_in = sizes[2];
    int H_in = sizes[3];
    int W_in = sizes[4];

    int D_out = (D_in + stride_d - 1) / stride_d;
    int H_out = (H_in + stride_h - 1) / stride_h;
    int W_out = (W_in + stride_w - 1) / stride_w;

    auto output = torch::empty({N, C, D_out, H_out, W_out}, input.options());
    int total_out_size = N * C * D_out * H_out * W_out;

    int block_size = 256;
    int grid_size = (total_out_size + block_size - 1) / block_size;

    avg_pool_3d_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D_in, H_in, W_in,
        D_out, H_out, W_out,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w
    );

    return output;
}
"""

cpp_src = r"""
torch::Tensor average_pool3d_cuda(torch::Tensor input, int kernel_d, int kernel_h, int kernel_w);
"""

# Load and compile the custom CUDA average pooling
custom_3d_avgpool = load_inline(
    name="custom_3d_avgpool",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["average_pool3d_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    A model that performs a 3D transposed convolution, followed by batch normalization,
    then two custom 3D average poolings (kernel=2).
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                                 stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.custom_3d_avgpool = custom_3d_avgpool

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        x = self.custom_3d_avgpool.average_pool3d_cuda(x, 2, 2, 2)
        x = self.custom_3d_avgpool.average_pool3d_cuda(x, 2, 2, 2)
        return x

def get_inputs():
    batch_size = 64
    in_channels = 3
    depth, height, width = 32, 32, 32
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    in_channels = 3
    out_channels = 16
    kernel_size = 3
    stride = 2
    padding = 1
    bias_shape = (16, 1, 1, 1)
    return [in_channels, out_channels, kernel_size, stride, padding, bias_shape]
