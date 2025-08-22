import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ----------------------------- CUDA source -----------------------------
source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_CUDA(x)       TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

// Kernel: fused 2D transposed convolution + bias addition + tanh
__global__ void conv_transpose2d_bias_tanh_kernel(
        const float* __restrict__ inp,      // [N, Cin, Hin, Win]
        const float* __restrict__ weight,   // [Cin, Cout, kH, kW]
        const float* __restrict__ bias,     // [Cout]
        float* __restrict__ out,            // [N, Cout, Hout, Wout]
        int N, int Cin, int Hin, int Win,
        int Cout, int Hout, int Wout,
        int kH, int kW,
        int stride, int padding, int output_padding) {

    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)N * Cout * Hout * Wout;
    if (tid >= total) return;

    // Unravel flattened index
    int ow = tid % Wout;
    int oh = (tid / Wout) % Hout;
    int oc = (tid / (Wout * Hout)) % Cout;
    int n  = tid / (Wout * Hout * Cout);

    float sum = 0.0f;

    // For each input channel and kernel element
    for (int ic = 0; ic < Cin; ++ic) {
        for (int kh = 0; kh < kH; ++kh) {
            int ih_numer = oh + padding - (kH - 1 - kh);
            if (ih_numer < 0) continue;
            if (ih_numer % stride) continue;
            int ih = ih_numer / stride;
            if (ih >= Hin) continue;

            for (int kw = 0; kw < kW; ++kw) {
                int iw_numer = ow + padding - (kW - 1 - kw);
                if (iw_numer < 0) continue;
                if (iw_numer % stride) continue;
                int iw = iw_numer / stride;
                if (iw >= Win) continue;

                long inp_idx = (((long)n * Cin + ic) * Hin + ih) * Win + iw;
                // Flip kernel spatially for transposed convolution
                int kh_flipped = kH - 1 - kh;
                int kw_flipped = kW - 1 - kw;
                long w_idx   = (((long)ic * Cout + oc) * kH + kh_flipped) * kW + kw_flipped;

                sum += inp[inp_idx] * weight[w_idx];
            }
        }
    }

    // Apply bias (addition), tanh and write
    sum += bias[oc];
    sum = tanhf(sum);

    out[tid] = sum;
}

torch::Tensor conv_transpose2d_bias_tanh_cuda(torch::Tensor input,
                                              torch::Tensor weight,
                                              torch::Tensor bias,
                                              int64_t stride,
                                              int64_t padding,
                                              int64_t output_padding) {
    // Sanity checks
    CHECK_CUDA(input);
    CHECK_CUDA(weight);
    CHECK_CUDA(bias);
    CHECK_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(weight);
    CHECK_CONTIGUOUS(bias);
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Only float32 supported");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "Only float32 supported");
    TORCH_CHECK(bias.dtype() == torch::kFloat32,  "Only float32 supported");

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
    torch::Tensor output = torch::empty({N, Cout, Hout, Wout}, opts);

    const long total = (long)N * Cout * Hout * Wout;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    conv_transpose2d_bias_tanh_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, Cin, Hin, Win,
        Cout, Hout, Wout,
        kH, kW,
        (int)stride, (int)padding, (int)output_padding);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel failed : ") + cudaGetErrorString(err));
    }

    return output;
}
"""

# ------------------------ C++ prototype declarations -------------------
cpp_src = """
torch::Tensor conv_transpose2d_bias_tanh_cuda(torch::Tensor input,
                                              torch::Tensor weight,
                                              torch::Tensor bias,
                                              int64_t stride,
                                              int64_t padding,
                                              int64_t output_padding);
"""

# ------------------------- Compile & load kernel ------------------------
conv_transpose2d = load_inline(
    name="conv_transpose2d_bias_tanh",
    cpp_sources=[cpp_src],
    cuda_sources=[source],
    functions=["conv_transpose2d_bias_tanh_cuda"],
    verbose=False,
)

# ------------------------------ ModelNew --------------------------------
class ModelNew(nn.Module):
    """
    Optimised model using a fused custom CUDA kernel to perform
    transposed convolution, bias addition, and tanh activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape,
                 stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, kernel_size, kernel_size, device="cuda", dtype=torch.float32))
        self.bias   = nn.Parameter(torch.randn(bias_shape, device="cuda", dtype=torch.float32))

        self.stride = int(stride)
        self.padding = int(padding)
        self.output_padding = int(output_padding)

    def forward(self, x):
        if not x.is_cuda:
            raise RuntimeError("Input must be a CUDA tensor.")
        return conv_transpose2d.conv_transpose2d_bias_tanh_cuda(
            x.contiguous(), self.weight.contiguous(), self.bias.contiguous(),
            self.stride, self.padding, self.output_padding)
