import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# CUDA source: 3-D transposed convolution fused with successive element-wise ops
# Implements:
#     y0  = conv_transpose3d(x, weight, bias)
#     out = 2*y0*y0 + y0*(bias + 1)
# -----------------------------------------------------------------------------
source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)  \
  CHECK_CUDA(x);        \
  CHECK_CONTIGUOUS(x)

////////////////////////////////////////////////////////////////////////
// Kernel
////////////////////////////////////////////////////////////////////////
__global__ void transposed_conv3d_fused_kernel(
        const float* __restrict__ input,    // [N, Cin,  Din, Hin, Win]
        const float* __restrict__ weight,   // [Cin, Cout, kD, kH, kW]
        const float* __restrict__ bias,     // [Cout]
        float*       __restrict__ output,   // [N, Cout, Dout, Hout, Wout]
        const int N,   const int Cin,
        const int Din, const int Hin, const int Win,
        const int Cout,
        const int kD,  const int kH,  const int kW,
        const int stride,
        const int padding,
        const int out_pad,
        const int Dout, const int Hout, const int Wout)
{
    const int total = N * Cout * Dout * Hout * Wout;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    /* Decode linear index -> (n, co, d_out, h_out, w_out) */
    int w_out = idx % Wout;
    int tmp   = idx / Wout;
    int h_out = tmp % Hout;
    tmp      /= Hout;
    int d_out = tmp % Dout;
    tmp      /= Dout;
    int co    = tmp % Cout;
    int n     = tmp / Cout;

    float val = 0.0f;

    /* Iterate over input channels & kernel elements */
    for (int ci = 0; ci < Cin; ++ci) {
        for (int kd = 0; kd < kD; ++kd) {
            int in_d_temp = d_out + padding - kd;
            if (in_d_temp % stride != 0) continue;
            int in_d = in_d_temp / stride;
            if (in_d < 0 || in_d >= Din) continue;

            for (int kh = 0; kh < kH; ++kh) {
                int in_h_temp = h_out + padding - kh;
                if (in_h_temp % stride != 0) continue;
                int in_h = in_h_temp / stride;
                if (in_h < 0 || in_h >= Hin) continue;

                for (int kw = 0; kw < kW; ++kw) {
                    int in_w_temp = w_out + padding - kw;
                    if (in_w_temp % stride != 0) continue;
                    int in_w = in_w_temp / stride;
                    if (in_w < 0 || in_w >= Win) continue;

                    /* Input index */
                    const int inp_idx =
                        (((n * Cin + ci) * Din + in_d) * Hin + in_h) * Win + in_w;

                    /* Weight index – NO flipping for conv_transpose */
                    const int w_idx =
                        ((((ci * Cout + co) * kD + kd) * kH + kh) * kW + kw);

                    val += input[inp_idx] * weight[w_idx];
                }
            }
        }
    }

    /* Add bias, apply fused element-wise ops */
    const float b  = bias[co];
    const float y0 = val + b;
    const float out_val = 2.f * y0 * y0 + y0 * (b + 1.f);

    /* Store */
    const int out_idx =
        (((n * Cout + co) * Dout + d_out) * Hout + h_out) * Wout + w_out;
    output[out_idx] = out_val;
}

////////////////////////////////////////////////////////////////////////
// C++ launcher
////////////////////////////////////////////////////////////////////////
torch::Tensor transposed_conv3d_fused_cuda(
        torch::Tensor input,     // [N, Cin, Din, Hin, Win]
        torch::Tensor weight,    // [Cin, Cout, kD, kH, kW]
        torch::Tensor bias,      // [Cout]
        int stride,
        int padding,
        int output_padding)
{
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);

    const int N    = input.size(0);
    const int Cin  = input.size(1);
    const int Din  = input.size(2);
    const int Hin  = input.size(3);
    const int Win  = input.size(4);

    const int kD   = weight.size(2);
    const int kH   = weight.size(3);
    const int kW   = weight.size(4);
    const int Cout = weight.size(1);

    /* Output sizes (same formula as PyTorch) */
    const int Dout = (Din - 1) * stride - 2 * padding + kD + output_padding;
    const int Hout = (Hin - 1) * stride - 2 * padding + kH + output_padding;
    const int Wout = (Win - 1) * stride - 2 * padding + kW + output_padding;

    auto options = input.options();
    torch::Tensor output = torch::empty({N, Cout, Dout, Hout, Wout}, options);

    const int total   = N * Cout * Dout * Hout * Wout;
    const int threads = 256;
    const int blocks  = (total + threads - 1) / threads;

    transposed_conv3d_fused_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, Cin,
        Din, Hin, Win,
        Cout,
        kD, kH, kW,
        stride,
        padding,
        output_padding,
        Dout, Hout, Wout);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        AT_ERROR("CUDA kernel failed : ", cudaGetErrorString(err));

    return output;
}
"""

# ---------------------------------------------------------------------
# C++ function prototypes
# ---------------------------------------------------------------------
cpp_src = """
torch::Tensor transposed_conv3d_fused_cuda(torch::Tensor input,
                                           torch::Tensor weight,
                                           torch::Tensor bias,
                                           int stride,
                                           int padding,
                                           int output_padding);
"""

# ---------------------------------------------------------------------
# Compile & load
# ---------------------------------------------------------------------
fused_conv = load_inline(
    name="fused_transposed_conv3d",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["transposed_conv3d_fused_cuda"],
    verbose=False,
)

# ---------------------------------------------------------------------
# PyTorch module using the custom kernel
# ---------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Performs:
        y0 = conv_transpose3d(x, weight, bias)
        out = 2*y0*y0 + y0*(bias + 1)
    (fused in custom CUDA kernel)
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 output_padding,
                 bias_shape):
        super().__init__()
        if isinstance(kernel_size, int):
            kD = kH = kW = kernel_size
        else:
            kD, kH, kW = kernel_size
        weight = torch.randn(in_channels, out_channels, kD, kH, kW)
        bias   = torch.randn(bias_shape)
        self.weight = nn.Parameter(weight)
        self.bias   = nn.Parameter(bias.view(-1))
        self.stride         = stride if isinstance(stride, int) else stride[0]
        self.padding        = padding if isinstance(padding, int) else padding[0]
        self.output_padding = output_padding if isinstance(output_padding, int) else output_padding[0]

    def forward(self, x):
        x      = x.contiguous().cuda()
        weight = self.weight.contiguous().cuda()
        bias   = self.bias.contiguous().cuda()
        return fused_conv.transposed_conv3d_fused_cuda(
            x, weight, bias,
            self.stride,
            self.padding,
            self.output_padding
        )
