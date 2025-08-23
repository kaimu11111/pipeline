import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_subt_tanh_subt_avgpool_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int N, int C, int H, int W,
    float subtract1_value,
    float subtract2_value,
    int kernel_size_pool)
{
    int H_out = H / kernel_size_pool;
    int W_out = W / kernel_size_pool;
    int total_out_elems = N * C * H_out * W_out;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_out_elems) {
        int temp = idx;
        int ow = temp % W_out; 
        temp /= W_out;
        int oh = temp % H_out; 
        temp /= H_out;
        int c_ = temp % C; 
        int n_ = temp / C;

        float sum_val = 0.0f;
        for(int kh = 0; kh < kernel_size_pool; kh++){
            for(int kw = 0; kw < kernel_size_pool; kw++){
                int h_in = oh * kernel_size_pool + kh;
                int w_in = ow * kernel_size_pool + kw;
                sum_val += x[((n_ * C + c_) * H + h_in) * W + w_in];
            }
        }
        float avg_val = sum_val / (float)(kernel_size_pool * kernel_size_pool);
        float val_sub1 = avg_val - subtract1_value;
        float val_tanh = tanhf(val_sub1);
        float val_sub2 = val_tanh - subtract2_value;
        out[((n_ * C + c_) * H_out + oh) * W_out + ow] = val_sub2;
    }
}

torch::Tensor fused_subt_tanh_subt_avgpool_cuda(
    torch::Tensor x,
    float subtract1_value,
    float subtract2_value,
    int kernel_size_pool)
{
    TORCH_CHECK(x.dim() == 4, "Input tensor x must have 4 dimensions (N, C, H, W)");
    TORCH_CHECK(x.dtype() == torch::kFloat, "Input tensor x must be float32");
    TORCH_CHECK(x.is_cuda(), "Input tensor x must be on CUDA device");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    int H_out = H / kernel_size_pool;
    int W_out = W / kernel_size_pool;

    auto out = torch::empty({N, C, H_out, W_out}, x.options());
    int total_out_elems = N * C * H_out * W_out;

    const int block_size = 256;
    const int grid_size = (total_out_elems + block_size - 1) / block_size;

    fused_subt_tanh_subt_avgpool_kernel<<<grid_size, block_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, H, W,
        subtract1_value, subtract2_value,
        kernel_size_pool
    );

    return out;
}
"""

cpp_src = r"""
torch::Tensor fused_subt_tanh_subt_avgpool_cuda(
    torch::Tensor x,
    float subtract1_value,
    float subtract2_value,
    int kernel_size_pool);
"""

fused_subt_tanh_subt_avgpool = load_inline(
    name="fused_subt_tanh_subt_avgpool",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["fused_subt_tanh_subt_avgpool_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    """
    Optimized Model that delegates subtract->tanh->subtract->avgpool to a fused custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.kernel_size_pool = kernel_size_pool
        self.fused_op = fused_subt_tanh_subt_avgpool

    def forward(self, x):
        x = self.conv(x)
        # Fused custom kernel handles subtract1 -> tanh -> subtract2 -> avgpool
        x = self.fused_op.fused_subt_tanh_subt_avgpool_cuda(
            x,
            self.subtract1_value,
            self.subtract2_value,
            self.kernel_size_pool
        )
        return x

batch_size = 128
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3
subtract1_value = 0.5
subtract2_value = 0.2
kernel_size_pool = 2

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool]
