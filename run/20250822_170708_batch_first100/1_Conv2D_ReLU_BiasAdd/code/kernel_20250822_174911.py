import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void conv_relu_bias_fused_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ conv_bias,
    const float* __restrict__ additional_bias,
    float* __restrict__ out,
    const int N,
    const int C_in,
    const int H,
    const int W,
    const int C_out,
    const int K,
    const int H_out,
    const int W_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * H_out * W_out;
    if (idx < total)
    {
        int w_out = idx % W_out;
        int h_out = (idx / W_out) % H_out;
        int c_out = (idx / (W_out * H_out)) % C_out;
        int n     = idx / (W_out * H_out * C_out);

        float val = 0.0f;
        // Convolution
        for(int c_in = 0; c_in < C_in; c_in++){
            for(int kh = 0; kh < K; kh++){
                for(int kw = 0; kw < K; kw++){
                    int h_in = h_out + kh;
                    int w_in = w_out + kw;
                    float inp_val = input[n*C_in*H*W + c_in*H*W + h_in*W + w_in];
                    float wgt_val = weight[c_out*C_in*K*K + c_in*K*K + kh*K + kw];
                    val += inp_val * wgt_val;
                }
            }
        }
        // Add conv bias
        val += conv_bias[c_out];
        // ReLU
        val = (val > 0.0f) ? val : 0.0f;
        // Add additional bias
        val += additional_bias[c_out];

        out[idx] = val;
    }
}

torch::Tensor fused_conv_relu_bias_add_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor conv_bias,
    torch::Tensor additional_bias)
{
    // input shape: (N, C_in, H, W)
    // weight shape: (C_out, C_in, K, K)
    // conv_bias shape: (C_out)
    // additional_bias shape: (C_out, 1, 1)
    
    const auto N = input.size(0);
    const auto C_in = input.size(1);
    const auto H = input.size(2);
    const auto W = input.size(3);

    const auto C_out = weight.size(0);
    const auto K = weight.size(2);

    // No padding, stride=1
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto out = torch::empty({N, C_out, H_out, W_out}, options);

    int total = N * C_out * H_out * W_out;
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;

    conv_relu_bias_fused_kernel<<<gridSize, blockSize>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        conv_bias.data_ptr<float>(),
        additional_bias.data_ptr<float>(),
        out.data_ptr<float>(),
        N,
        C_in,
        H,
        W,
        C_out,
        K,
        H_out,
        W_out
    );

    return out;
}
"""

cpp_src = r"""
torch::Tensor fused_conv_relu_bias_add_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor conv_bias,
    torch::Tensor additional_bias);
"""

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["fused_conv_relu_bias_add_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    """
    Optimized model that fuses convolution, ReLU, and bias addition.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        # Replicate the original parameter shapes
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.conv_bias = nn.Parameter(torch.randn(out_channels))
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        return fused_ops.fused_conv_relu_bias_add_cuda(
            x, self.weight, self.conv_bias, self.bias
        )

batch_size = 64
in_channels = 64
out_channels = 128
height = width = 128
kernel_size = 3
bias_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]
