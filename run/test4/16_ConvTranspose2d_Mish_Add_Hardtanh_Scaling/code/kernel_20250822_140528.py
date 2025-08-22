import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -------------------------------- CUDA source ---------------------------------
source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CUDA_1D_KERNEL_LOOP(i, n)                              \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
         i += blockDim.x * gridDim.x)

__global__ void conv_transpose2d_fused_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int N,
    const int C_in,
    const int H_in,
    const int W_in,
    const int C_out,
    const int H_out,
    const int W_out,
    const int K_h,
    const int K_w,
    const int stride,
    const int padding,
    const int output_padding,
    const float add_value,
    const float scale)
{
    const int total = N * C_out * H_out * W_out;
    CUDA_1D_KERNEL_LOOP(idx, total) {
        // Decode indices
        int tmp = idx;
        const int w_out = tmp % W_out;
        tmp /= W_out;
        const int h_out = tmp % H_out;
        tmp /= H_out;
        const int c_out = tmp % C_out;
        const int n = tmp / C_out;

        float sum = 0.0f;

        // Iterate over input channels and kernel
        for (int c_in = 0; c_in < C_in; ++c_in) {
            for (int k_h = 0; k_h < K_h; ++k_h) {
                int h_in_nom = h_out + padding - k_h;
                if (h_in_nom % stride != 0) continue;
                int h_in = h_in_nom / stride;
                if (h_in < 0 || h_in >= H_in) continue;

                for (int k_w = 0; k_w < K_w; ++k_w) {
                    int w_in_nom = w_out + padding - k_w;
                    if (w_in_nom % stride != 0) continue;
                    int w_in = w_in_nom / stride;
                    if (w_in < 0 || w_in >= W_in) continue;

                    // Input index: (((n*C_in + c_in)*H_in + h_in)*W_in + w_in)
                    int in_idx = (((n * C_in + c_in) * H_in + h_in) * W_in + w_in);
                    // Weight index: (((c_in*C_out + c_out)*K_h + k_h)*K_w + k_w)
                    int w_idx = (((c_in * C_out + c_out) * K_h + k_h) * K_w + k_w);
                    sum += input[in_idx] * weight[w_idx];
                }
            }
        }

        // Mish activation
        float sp = logf(1.0f + expf(sum));
        float mish_val = sum * tanhf(sp);

        // Add, Hardtanh, scale
        float out_val = mish_val + add_value;
        if (out_val < -1.0f) out_val = -1.0f;
        else if (out_val > 1.0f) out_val = 1.0f;
        out_val *= scale;

        output[idx] = out_val;
    }
}

torch::Tensor conv_transpose2d_fused_cuda(torch::Tensor input,
                                          torch::Tensor weight,
                                          int stride,
                                          int padding,
                                          int output_padding,
                                          float add_value,
                                          float scale) {
    // Ensure tensors are on CUDA and contiguous
    input = input.contiguous();
    weight = weight.contiguous();

    const int N = input.size(0);
    const int C_in = input.size(1);
    const int H_in = input.size(2);
    const int W_in = input.size(3);

    const int C_out = weight.size(1);
    const int K_h   = weight.size(2);
    const int K_w   = weight.size(3);

    const int H_out = (H_in - 1) * stride - 2 * padding + K_h + output_padding;
    const int W_out = (W_in - 1) * stride - 2 * padding + K_w + output_padding;

    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    torch::Tensor output = torch::empty({N, C_out, H_out, W_out}, options);

    const int total = N * C_out * H_out * W_out;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    conv_transpose2d_fused_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        K_h, K_w,
        stride, padding, output_padding,
        add_value, scale);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel failed : %s\n", cudaGetErrorString(err));
    }
    return output;
}
"""

cpp_src = """
torch::Tensor conv_transpose2d_fused_cuda(torch::Tensor input,
                                          torch::Tensor weight,
                                          int stride,
                                          int padding,
                                          int output_padding,
                                          float add_value,
                                          float scale);
"""

# ------------------------------- load extension -------------------------------
conv_transpose2d_fused = load_inline(
    name="conv_transpose2d_fused",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["conv_transpose2d_fused_cuda"],
    verbose=False,
)

# --------------------------------- ModelNew -----------------------------------
class ModelNew(nn.Module):
    """
    Optimised model replacing transposed convolution, Mish, add, Hardtanh and scaling
    with a single fused CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 output_padding, add_value, scale):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels,
                                               kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.stride = int(stride)
        self.padding = int(padding)
        self.output_padding = int(output_padding)
        self.add_value = float(add_value)
        self.scale = float(scale)

    def forward(self, x):
        return conv_transpose2d_fused.conv_transpose2d_fused_cuda(
            x,
            self.weight,
            self.stride,
            self.padding,
            self.output_padding,
            self.add_value,
            self.scale
        )
