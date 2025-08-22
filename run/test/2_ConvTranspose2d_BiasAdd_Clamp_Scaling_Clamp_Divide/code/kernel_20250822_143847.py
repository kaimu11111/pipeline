import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------
# 1. CUDA / C++ source code
# ------------------------------------------------------------
source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x)       TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT32(x)    TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")

// -----------------------------------------------------------------------------
// CUDA kernel : fused conv_transpose2d + bias + clamp/scale/clamp/div
// -----------------------------------------------------------------------------
__global__ void conv_transpose2d_fused_kernel(
        const float* __restrict__ inp,     // [N, Cin, Hin, Win]
        const float* __restrict__ weight,  // [Cin, Cout, Kh, Kw]
        const float* __restrict__ bias,    // [Cout]
        float* __restrict__ out,           // [N, Cout, Hout, Wout]
        int N, int Cin, int Hin, int Win,
        int Cout, int Kh, int Kw,
        int stride, int pad,
        int Hout, int Wout,
        float scaling)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * Cout * Hout * Wout;
    if (idx >= total) return;

    // Decode flat index -> n, co, h_out, w_out
    int w_out = idx % Wout;
    int tmp   = idx / Wout;
    int h_out = tmp % Hout;
    tmp      /= Hout;
    int co    = tmp % Cout;
    int n     = tmp / Cout;

    float acc = bias[co];

    for (int ci = 0; ci < Cin; ++ci){
        const float* weight_ci = weight + ((ci * Cout + co) * Kh * Kw); // [Kh, Kw] slice

        for (int kh = 0; kh < Kh; ++kh){
            // Correct mapping from output index to input index for transposed convolution
            int h_in_nom = h_out + pad - kh;
            if (h_in_nom < 0) continue;
            if (h_in_nom % stride != 0) continue;
            int h_in = h_in_nom / stride;
            if (h_in < 0 || h_in >= Hin) continue;

            for (int kw = 0; kw < Kw; ++kw){
                int w_in_nom = w_out + pad - kw;
                if (w_in_nom < 0) continue;
                if (w_in_nom % stride != 0) continue;
                int w_in = w_in_nom / stride;
                if (w_in < 0 || w_in >= Win) continue;

                float inp_val = inp[(((n * Cin + ci) * Hin + h_in) * Win + w_in)];
                float w_val   = weight_ci[kh * Kw + kw];
                acc += inp_val * w_val;
            }
        }
    }

    // Fused activation / scaling
    acc = fminf(fmaxf(acc, 0.0f), 1.0f); // clamp 0..1
    acc *= scaling;
    acc = fminf(fmaxf(acc, 0.0f), 1.0f); // clamp 0..1
    acc /= scaling;

    out[idx] = acc;
}

// -----------------------------------------------------------------------------
// Host launcher
// -----------------------------------------------------------------------------
torch::Tensor conv_transpose2d_fused_cuda(
        torch::Tensor input,        // [N, Cin, Hin, Win]
        torch::Tensor weight,       // [Cin, Cout, Kh, Kw]
        torch::Tensor bias,         // [Cout]
        int stride,
        int padding,
        float scaling_factor,
        int output_padding)
{
    // Sanity checks
    CHECK_CUDA(input);
    CHECK_CUDA(weight);
    CHECK_CUDA(bias);
    CHECK_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(weight);
    CHECK_CONTIGUOUS(bias);
    CHECK_FLOAT32(input);
    CHECK_FLOAT32(weight);
    CHECK_FLOAT32(bias);

    const int N   = input.size(0);
    const int Cin = input.size(1);
    const int Hin = input.size(2);
    const int Win = input.size(3);

    const int Kh  = weight.size(2);
    const int Kw  = weight.size(3);
    const int Cout= weight.size(1);

    const int Hout = (Hin - 1) * stride - 2 * padding + Kh + output_padding;
    const int Wout = (Win - 1) * stride - 2 * padding + Kw + output_padding;

    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto output  = torch::empty({N, Cout, Hout, Wout}, options);

    const int threads = 256;
    const int total   = N * Cout * Hout * Wout;
    const int blocks  = (total + threads - 1) / threads;

    conv_transpose2d_fused_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, Cin, Hin, Win,
        Cout, Kh, Kw,
        stride, padding,
        Hout, Wout,
        scaling_factor
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        TORCH_CHECK(false, "CUDA kernel launch error : ", cudaGetErrorString(err));
    }

    return output;
}
"""

# ------------------------------------------------------------
# 2. C++ prototypes
# ------------------------------------------------------------
cpp_src = """
torch::Tensor conv_transpose2d_fused_cuda(
        torch::Tensor input,
        torch::Tensor weight,
        torch::Tensor bias,
        int stride,
        int padding,
        float scaling_factor,
        int output_padding);
"""

# ------------------------------------------------------------
# 3. Compile / load the CUDA extension
# ------------------------------------------------------------
fused_conv = load_inline(
    name="fused_conv_transpose2d_fixed",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["conv_transpose2d_fused_cuda"],
    verbose=False,
)

# ------------------------------------------------------------
# 4. Replacement Model
# ------------------------------------------------------------
class ModelNew(nn.Module):
    """
    PyTorch Module with all core operators replaced by a single fused CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, output_padding, bias_shape, scaling_factor):
        super().__init__()
        weight = torch.randn(in_channels, out_channels, kernel_size, kernel_size)
        self.weight = nn.Parameter(weight)
        self.bias   = nn.Parameter(torch.randn(*bias_shape))
        self.stride = int(stride)
        self.padding = int(padding)
        self.output_padding = int(output_padding)
        self.scaling_factor = float(scaling_factor)

    def forward(self, x):
        return fused_conv.conv_transpose2d_fused_cuda(
            x.contiguous(),
            self.weight.contiguous(),
            self.bias.view(-1).contiguous(),
            self.stride,
            self.padding,
            self.scaling_factor,
            self.output_padding
        )
