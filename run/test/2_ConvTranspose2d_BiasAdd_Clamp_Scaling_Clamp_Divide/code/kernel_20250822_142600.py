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
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT32(x) TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")

// -----------------------------------------------------------------------------
// Utility: compute output spatial size for transposed convolution
// -----------------------------------------------------------------------------
inline int out_size(int in_size, int kernel, int stride, int pad, int out_pad){
    return (in_size - 1) * stride - 2 * pad + kernel + out_pad;
}

// -----------------------------------------------------------------------------
// CUDA kernel: fused transposed conv2d + bias + clamp/scale/clamp/div
// -----------------------------------------------------------------------------
__global__ void conv_transpose2d_fused_kernel(
        const float* __restrict__ inp,     // [N, Cin, Hin, Win]
        const float* __restrict__ weight,  // [Cin, Cout, Kh, Kw]
        const float* __restrict__ bias,    // [Cout]
        float* __restrict__ out,           // [N, Cout, Hout, Wout]
        int N, int Cin, int Hin, int Win,
        int Cout, int Kh, int Kw,
        int stride, int pad, int out_pad,
        int Hout, int Wout,
        float scaling)
{
    // total elements
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * Cout * Hout * Wout;
    if (idx >= total) return;

    // Decode flat index -> n, co, h, w
    int w_out = idx % Wout;
    int tmp = idx / Wout;
    int h_out = tmp % Hout;
    tmp /= Hout;
    int co = tmp % Cout;
    int n = tmp / Cout;

    float acc = bias[co];

    // Iterate over input channels and kernel
    for (int ci = 0; ci < Cin; ++ci){
        const float* weight_ci = weight + (((ci * Cout) + co) * Kh * Kw); // pointer to [Kh, Kw]
        // For each kernel position, determine if it maps to a valid input position
        for (int kh = 0; kh < Kh; ++kh){
            // Solve for input h: h_out = (h_in - 1) * stride - 2*pad + kh + out_pad_cond
            int h_in_nom = h_out + pad - kh;
            if (h_in_nom % stride != 0) continue;
            int h_in = h_in_nom / stride;
            if (h_in < 0 || h_in >= Hin) continue;

            for (int kw = 0; kw < Kw; ++kw){
                int w_in_nom = w_out + pad - kw;
                if (w_in_nom % stride != 0) continue;
                int w_in = w_in_nom / stride;
                if (w_in < 0 || w_in >= Win) continue;

                float inp_val = inp[ (((n * Cin + ci) * Hin + h_in) * Win + w_in) ];
                float w_val   = weight_ci[kh * Kw + kw];
                acc += inp_val * w_val;
            }
        }
    }

    // Fused activation / scaling
    acc = fminf(fmaxf(acc, 0.0f), 1.0f);     // clamp 0..1
    acc *= scaling;
    acc = fminf(fmaxf(acc, 0.0f), 1.0f);     // clamp again
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
        int output_padding,
        float scaling_factor)
{
    // Input checks
    CHECK_CUDA(input);
    CHECK_CUDA(weight);
    CHECK_CUDA(bias);
    CHECK_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(weight);
    CHECK_CONTIGUOUS(bias);
    CHECK_FLOAT32(input);
    CHECK_FLOAT32(weight);
    CHECK_FLOAT32(bias);

    auto N    = input.size(0);
    auto Cin  = input.size(1);
    auto Hin  = input.size(2);
    auto Win  = input.size(3);

    auto Kh = weight.size(2);
    auto Kw = weight.size(3);
    auto Cout = weight.size(1);

    int Hout = (Hin - 1) * stride - 2 * padding + Kh + output_padding;
    int Wout = (Win - 1) * stride - 2 * padding + Kw + output_padding;

    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto output  = torch::zeros({N, Cout, Hout, Wout}, options);

    const int threads = 256;
    int total = N * Cout * Hout * Wout;
    int blocks = (total + threads - 1) / threads;

    conv_transpose2d_fused_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, Cin, Hin, Win,
        Cout, Kh, Kw,
        stride, padding, output_padding,
        Hout, Wout,
        scaling_factor
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("CUDA kernel failed : %s\n", cudaGetErrorString(err));
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_transpose2d_fused_cuda", &conv_transpose2d_fused_cuda, "Fused Transposed Conv2d");
}
"""

cpp_src = """
torch::Tensor conv_transpose2d_fused_cuda(
        torch::Tensor input,
        torch::Tensor weight,
        torch::Tensor bias,
        int stride,
        int padding,
        int output_padding,
        float scaling_factor);
"""

# ------------------------------------------------------------
# 2. Compile / load the CUDA extension
# ------------------------------------------------------------
fused_conv = load_inline(
    name="fused_conv_transpose2d",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["conv_transpose2d_fused_cuda"],
    verbose=False,
)

# ------------------------------------------------------------
# 3. Replacement Model
# ------------------------------------------------------------
class ModelNew(nn.Module):
    """
    PyTorch Module with all core operators replaced by a single fused CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, output_padding, bias_shape, scaling_factor):
        super().__init__()
        # Register weights & bias to keep training compatibility
        weight = torch.randn(in_channels, out_channels, kernel_size, kernel_size)  # same init as default
        self.weight = nn.Parameter(weight)
        self.bias   = nn.Parameter(torch.randn(*bias_shape))
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.scaling_factor = scaling_factor

    def forward(self, x):
        return fused_conv.conv_transpose2d_fused_cuda(
            x,
            self.weight,
            self.bias.view(-1),  # (Cout)
            self.stride,
            self.padding,
            self.output_padding,
            float(self.scaling_factor)
        )
