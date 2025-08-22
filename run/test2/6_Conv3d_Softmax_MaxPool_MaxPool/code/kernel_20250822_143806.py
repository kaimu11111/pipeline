import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# --------------------------------------------------------------------
# CUDA kernels and wrappers (no explicit PYBIND11_MODULE to avoid
# duplicate definition – torch will autogenerate it from `functions=`)
# --------------------------------------------------------------------
source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

////////////////////////////////////////////////////////////////
// Helpers
////////////////////////////////////////////////////////////////
#define CUDA_1D_KERNEL_LOOP(i, n)                                           \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);              \
       i += blockDim.x * gridDim.x)

////////////////////////////////////////////////////////////////
// 3D Convolution (stride=1, padding=0, dilation=1)
////////////////////////////////////////////////////////////////
__global__ void conv3d_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in,
    int D_in, int H_in, int W_in,
    int C_out,
    int Kd, int Kh, int Kw,
    int D_out, int H_out, int W_out) {

    int total = N * C_out * D_out * H_out * W_out;
    CUDA_1D_KERNEL_LOOP(idx, total) {
        int w_out   = idx % W_out;
        int h_out   = (idx / W_out) % H_out;
        int d_out   = (idx / (W_out * H_out)) % D_out;
        int c_out   = (idx / (W_out * H_out * D_out)) % C_out;
        int n       = idx / (W_out * H_out * D_out * C_out);

        float val = bias[c_out];

        for (int c = 0; c < C_in; ++c) {
            for (int kd = 0; kd < Kd; ++kd) {
                int d_in = d_out + kd;
                for (int kh = 0; kh < Kh; ++kh) {
                    int h_in = h_out + kh;
                    for (int kw = 0; kw < Kw; ++kw) {
                        int w_in = w_out + kw;

                        int inp_idx = (((n * C_in + c) * D_in + d_in) * H_in + h_in) * W_in + w_in;
                        int wgt_idx = ((((c_out * C_in + c) * Kd + kd) * Kh + kh) * Kw) + kw;
                        val += input[inp_idx] * weight[wgt_idx];
                    }
                }
            }
        }
        int out_idx = idx;
        output[out_idx] = val;
    }
}

torch::Tensor conv3d_forward(
        torch::Tensor input,
        torch::Tensor weight,
        torch::Tensor bias) {

    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Only FP32 supported");

    int N  = input.size(0);
    int C_in = input.size(1);
    int D_in = input.size(2);
    int H_in = input.size(3);
    int W_in = input.size(4);

    int C_out = weight.size(0);
    int Kd = weight.size(2);
    int Kh = weight.size(3);
    int Kw = weight.size(4);

    int D_out = D_in - Kd + 1;
    int H_out = H_in - Kh + 1;
    int W_out = W_in - Kw + 1;

    auto output = torch::empty({N, C_out, D_out, H_out, W_out}, input.options());

    const int threads = 256;
    const int total = N * C_out * D_out * H_out * W_out;
    const int blocks = (total + threads - 1) / threads;

    conv3d_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in,
        D_in, H_in, W_in,
        C_out,
        Kd, Kh, Kw,
        D_out, H_out, W_out);

    return output;
}

////////////////////////////////////////////////////////////////
// Channel-wise Softmax
////////////////////////////////////////////////////////////////
__global__ void softmax_channel_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        int N, int C, int D, int H, int W) {

    int spatial = D * H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * spatial) return;

    int w = idx % W;
    int h = (idx / W) % H;
    int d = (idx / (W * H)) % D;
    int n = idx / (W * H * D);

    // compute max for numeric stability
    float max_val = -FLT_MAX;
    for (int c = 0; c < C; ++c) {
        int in_idx = ((((n * C + c) * D + d) * H + h) * W + w);
        max_val = fmaxf(max_val, input[in_idx]);
    }

    // compute exp and sum
    float sum_exp = 0.f;
    for (int c = 0; c < C; ++c) {
        int in_idx = ((((n * C + c) * D + d) * H + h) * W + w);
        float exp_val = expf(input[in_idx] - max_val);
        sum_exp += exp_val;
        output[in_idx] = exp_val; // temp store
    }

    // normalize
    for (int c = 0; c < C; ++c) {
        int out_idx = ((((n * C + c) * D + d) * H + h) * W + w);
        output[out_idx] /= sum_exp;
    }
}

torch::Tensor softmax_channel_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    int N = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);

    auto output = torch::empty_like(input);

    int threads = 256;
    int blocks = (N * D * H * W + threads - 1) / threads;

    softmax_channel_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D, H, W);

    return output;
}

////////////////////////////////////////////////////////////////
// MaxPool3d (cube kernel, stride == kernel)
////////////////////////////////////////////////////////////////
__global__ void maxpool3d_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        int N, int C,
        int D_in, int H_in, int W_in,
        int K,
        int D_out, int H_out, int W_out) {

    int total = N * C * D_out * H_out * W_out;
    CUDA_1D_KERNEL_LOOP(idx, total) {
        int w_out   = idx % W_out;
        int h_out   = (idx / W_out) % H_out;
        int d_out   = (idx / (W_out * H_out)) % D_out;
        int c       = (idx / (W_out * H_out * D_out)) % C;
        int n       = idx / (W_out * H_out * D_out * C);

        float max_val = -FLT_MAX;
        for (int kd = 0; kd < K; ++kd) {
            int d_in = d_out * K + kd;
            for (int kh = 0; kh < K; ++kh) {
                int h_in = h_out * K + kh;
                for (int kw = 0; kw < K; ++kw) {
                    int w_in = w_out * K + kw;
                    int in_idx = ((((n * C + c) * D_in + d_in) * H_in + h_in) * W_in + w_in);
                    max_val = fmaxf(max_val, input[inp_idx]);
                }
            }
        }
        int out_idx = idx;
        output[out_idx] = max_val;
    }
}

torch::Tensor maxpool3d_forward(torch::Tensor input, int kernel_size) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    int N = input.size(0);
    int C = input.size(1);
    int D_in = input.size(2);
    int H_in = input.size(3);
    int W_in = input.size(4);
    int K = kernel_size;

    int D_out = D_in / K;
    int H_out = H_in / K;
    int W_out = W_in / K;

    auto output = torch::empty({N, C, D_out, H_out, W_out}, input.options());

    int threads = 256;
    int total = N * C * D_out * H_out * W_out;
    int blocks = (total + threads - 1) / threads;

    maxpool3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C,
        D_in, H_in, W_in,
        K,
        D_out, H_out, W_out);

    return output;
}
"""

# Prototypes for functions exposed to Python
cpp_src = """
torch::Tensor conv3d_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
torch::Tensor softmax_channel_forward(torch::Tensor input);
torch::Tensor maxpool3d_forward(torch::Tensor input, int kernel_size);
"""

# Compile and load kernels
cuda_ops = load_inline(
    name="custom_cuda_ops",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=[
        "conv3d_forward",
        "softmax_channel_forward",
        "maxpool3d_forward",
    ],
    verbose=False,
)

# --------------------------------------------------------------------
# PyTorch module utilising the custom CUDA kernels
# --------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Optimised model with hand-written CUDA kernels for Conv3d, Softmax, and MaxPool3d.
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        self.pool_kernel_size = pool_kernel_size if isinstance(pool_kernel_size, int) else pool_kernel_size[0]

        # Parameters equivalent to nn.Conv3d
        weight_shape = (out_channels, in_channels, self.kernel_size, self.kernel_size, self.kernel_size)
        self.weight = nn.Parameter(torch.empty(weight_shape, dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty(out_channels, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = in_channels * self.kernel_size ** 3
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N, C_in, D, H, W) on CUDA device
        """
        if not x.is_cuda:
            x = x.cuda()
        weight = self.weight.cuda()
        bias = self.bias.cuda()

        x = cuda_ops.conv3d_forward(x, weight, bias)
        x = cuda_ops.softmax_channel_forward(x)
        x = cuda_ops.maxpool3d_forward(x, self.pool_kernel_size)
        x = cuda_ops.maxpool3d_forward(x, self.pool_kernel_size)
        return x
