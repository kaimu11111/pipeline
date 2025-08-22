import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ----------------------------------------------------------------------
# CUDA source
# ----------------------------------------------------------------------
source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_CUDA(x)       TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

// ------------------------------------------------------------------
// Kernel 1 : compute transposed convolution (no bias / activation)
// Each thread processes ONE input pixel (n, ic, ih, iw) and updates
// all affected output elements using atomicAdd.
// ------------------------------------------------------------------
__global__ void conv_transpose2d_kernel(
        const float* __restrict__ inp,      // [N, Cin, Hin, Win]
        const float* __restrict__ weight,   // [Cin, Cout, kH, kW]
        float* __restrict__ out,            // [N, Cout, Hout, Wout] – zero-initialised
        int N, int Cin, int Hin, int Win,
        int Cout, int Hout, int Wout,
        int kH, int kW,
        int stride, int padding, int /*output_padding (unused here)*/) {

    long tid  = blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)N * Cin * Hin * Win;
    if (tid >= total) return;

    // Decode input index
    int iw =  tid % Win;
    int ih = (tid / Win) % Hin;
    int ic = (tid / (Win * Hin)) % Cin;
    int n  =  tid / (Win * Hin * Cin);

    const float in_val = inp[tid];

    // Loop over kernel & output channels
    for (int oc = 0; oc < Cout; ++oc) {
        const float* w_ptr = weight + (((ic * Cout + oc) * kH) * kW); // pre-offset

        for (int kh = 0; kh < kH; ++kh) {
            int oh = ih * stride - padding + kh;
            if (oh < 0 || oh >= Hout) continue;

            for (int kw = 0; kw < kW; ++kw) {
                int ow = iw * stride - padding + kw;
                if (ow < 0 || ow >= Wout) continue;

                float w_val = w_ptr[kh * kW + kw];

                long out_idx = (((long)n * Cout + oc) * Hout + oh) * Wout + ow;
                atomicAdd(out + out_idx, in_val * w_val);
            }
        }
    }
}

// ------------------------------------------------------------------
// Kernel 2 : add bias and apply tanh activation
// ------------------------------------------------------------------
__global__ void bias_tanh_kernel(
        float* __restrict__ out,      // [N, Cout, Hout, Wout]
        const float* __restrict__ bias,
        long total_elements,
        int Cout) {

    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_elements) return;

    int oc = (tid / ((total_elements / Cout))) % Cout; // derive oc

    float val = out[tid] + bias[oc];
    out[tid] = tanhf(val);
}

// ------------------------------------------------------------------
// Host wrapper
// ------------------------------------------------------------------
torch::Tensor conv_transpose2d_bias_tanh_cuda(
        torch::Tensor input,
        torch::Tensor weight,
        torch::Tensor bias,
        int64_t stride,
        int64_t padding,
        int64_t output_padding) {

    // Basic checks
    CHECK_CUDA(input);   CHECK_CUDA(weight);   CHECK_CUDA(bias);
    CHECK_CONTIGUOUS(input); CHECK_CONTIGUOUS(weight); CHECK_CONTIGUOUS(bias);
    TORCH_CHECK(input.dtype()  == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype()   == torch::kFloat32, "bias must be float32");

    const int N   = input.size(0);
    const int Cin = input.size(1);
    const int Hin = input.size(2);
    const int Win = input.size(3);

    const int Cout = weight.size(1);
    const int kH   = weight.size(2);
    const int kW   = weight.size(3);

    const int Hout = (Hin - 1) * stride - 2 * padding + kH + output_padding;
    const int Wout = (Win - 1) * stride - 2 * padding + kW + output_padding;

    auto opts = input.options();
    torch::Tensor output = torch::zeros({N, Cout, Hout, Wout}, opts);

    // ------------------------------------------------------------------
    // Launch kernel 1
    // ------------------------------------------------------------------
    {
        const long total_inp = (long)N * Cin * Hin * Win;
        const int threads = 256;
        const int blocks  = (int)((total_inp + threads - 1) / threads);

        conv_transpose2d_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            output.data_ptr<float>(),
            N, Cin, Hin, Win,
            Cout, Hout, Wout,
            kH, kW,
            (int)stride, (int)padding, (int)output_padding);
    }

    // ------------------------------------------------------------------
    // Launch kernel 2 (bias + tanh)
    // ------------------------------------------------------------------
    {
        const long total_out = (long)N * Cout * Hout * Wout;
        const int threads = 256;
        const int blocks  = (int)((total_out + threads - 1) / threads);

        bias_tanh_kernel<<<blocks, threads>>>(
            output.data_ptr<float>(),
            bias.data_ptr<float>(),
            total_out,
            Cout);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel failed : ") + cudaGetErrorString(err));
    }

    return output;
}
"""

# ----------------------------------------------------------------------
# C++ prototypes
# ----------------------------------------------------------------------
cpp_src = r"""
torch::Tensor conv_transpose2d_bias_tanh_cuda(torch::Tensor input,
                                              torch::Tensor weight,
                                              torch::Tensor bias,
                                              int64_t stride,
                                              int64_t padding,
                                              int64_t output_padding);
"""

# ----------------------------------------------------------------------
# Compile & load
# ----------------------------------------------------------------------
conv_transpose2d = load_inline(
    name="conv_transpose2d_bias_tanh_fixed",
    cpp_sources=[cpp_src],
    cuda_sources=[source],
    functions=["conv_transpose2d_bias_tanh_cuda"],
    verbose=False,
)

# ----------------------------------------------------------------------
# Python module
# ----------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Optimised transposed-convolution layer: conv_transpose2d + bias + tanh
    implemented with a custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape,
                 stride=2, padding=1, output_padding=1):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, kernel_size, kernel_size,
                        device="cuda", dtype=torch.float32))
        self.bias = nn.Parameter(torch.randn(bias_shape, device="cuda", dtype=torch.float32))

        self.stride = int(stride)
        self.padding = int(padding)
        self.output_padding = int(output_padding)

    def forward(self, x):
        if not x.is_cuda:
            raise RuntimeError("Input must be a CUDA tensor.")
        return conv_transpose2d.conv_transpose2d_bias_tanh_cuda(
            x.contiguous(),
            self.weight.contiguous(),
            self.bias.contiguous(),
            self.stride,
            self.padding,
            self.output_padding)
