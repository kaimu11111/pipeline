import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA source for fused Conv2d + BiasAdd + ReLU (padding=0, stride=1, dilation=1, groups=1)
source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_relu_bias_kernel(
        const float* __restrict__ input,
        const float* __restrict__ weight,
        const float* __restrict__ bias,
        float* __restrict__ output,
        int N, int C_in, int C_out,
        int H, int W, int K) {

    long H_out = H - K + 1;
    long W_out = W - K + 1;
    long total = (long)N * C_out * H_out * W_out;

    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    long w_out = idx % W_out;
    long tmp = idx / W_out;
    long h_out = tmp % H_out;
    tmp /= H_out;
    long oc = tmp % C_out;
    long n  = tmp / C_out;

    float sum = 0.0f;

    for (int ic = 0; ic < C_in; ++ic) {
        for (int ky = 0; ky < K; ++ky) {
            int in_y = h_out + ky;
            for (int kx = 0; kx < K; ++kx) {
                int in_x = w_out + kx;

                long in_idx = (((n * C_in + ic) * H + in_y) * W + in_x);
                long w_idx  = (((oc * C_in + ic) * K + ky) * K + kx);

                sum += input[in_idx] * weight[w_idx];
            }
        }
    }

    sum += bias[oc];               // Add bias
    sum = fmaxf(sum, 0.0f);        // ReLU

    long out_idx = (((n * C_out + oc) * H_out + h_out) * W_out + w_out);
    output[out_idx] = sum;
}

torch::Tensor conv_relu_bias_cuda(torch::Tensor input,
                                  torch::Tensor weight,
                                  torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Only float32 supported");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "Only float32 supported");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "Only float32 supported");

    int N = input.size(0);
    int C_in = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int C_out = weight.size(0);
    int K = weight.size(2);

    int H_out = H - K + 1;
    int W_out = W - K + 1;

    auto output = torch::empty({N, C_out, H_out, W_out}, input.options());

    long total_elems = (long)N * C_out * H_out * W_out;
    const int threads = 256;
    const int blocks = (total_elems + threads - 1) / threads;

    conv_relu_bias_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, C_out,
        H, W, K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }

    return output;
}
"""

# C++ prototype for exposure
cpp_src = """
torch::Tensor conv_relu_bias_cuda(torch::Tensor input,
                                  torch::Tensor weight,
                                  torch::Tensor bias);
"""

# Compile and load the CUDA kernel
conv_relu_bias = load_inline(
    name="conv_relu_bias",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["conv_relu_bias_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Fused implementation of Conv2d (padding=0, stride=1) + ReLU + bias add.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        weight_shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.randn(weight_shape))
        # Store bias as (out_channels,) for simplicity; reshape at call time
        self.bias   = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        return conv_relu_bias.conv_relu_bias_cuda(
            x,
            self.weight,
            self.bias
        )
