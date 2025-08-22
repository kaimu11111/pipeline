import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------
# CUDA kernels (ConvTranspose3d + BatchNorm affine transform +
#               per-sample/channel spatial-mean subtraction)
# ---------------------------------------------------------------------
source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

////////////////////////////////////////////////////////////////////////////////
// Helpers
////////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ int div_up(int a, int b) { return (a + b - 1) / b; }

////////////////////////////////////////////////////////////////////////////////
// 1. ConvTranspose3d (NCDHW) – matches PyTorch semantic
//    NOTE: kernel indices are flipped compared with regular conv.
////////////////////////////////////////////////////////////////////////////////
__global__ void conv_transpose3d_kernel(
        const float *__restrict__ input,     // [N, C_in, D_in, H_in, W_in]
        const float *__restrict__ weight,    // [C_in, C_out, kD, kH, kW]
        const float *__restrict__ bias,      // [C_out]  (may be nullptr)
        float *__restrict__ output,          // [N, C_out, D_out, H_out, W_out]
        const int N, const int C_in,
        const int D_in, const int H_in, const int W_in,
        const int C_out,
        const int kD, const int kH, const int kW,
        const int stride, const int padding,
        const int D_out, const int H_out, const int W_out)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * C_out * D_out * H_out * W_out;
    if (tid >= total) return;

    int w_out =  tid % W_out;
    int h_out = (tid / W_out) % H_out;
    int d_out = (tid / (W_out * H_out)) % D_out;
    int c_out = (tid / (W_out * H_out * D_out)) % C_out;
    int n     =  tid / (W_out * H_out * D_out * C_out);

    float val = (bias != nullptr) ? bias[c_out] : 0.0f;

    // Loop over input channels and kernel elements
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int kd = 0; kd < kD; ++kd) {
            int d_in_nom = d_out + padding - kd;
            if (d_in_nom % stride != 0) continue;
            int d_in = d_in_nom / stride;
            if (d_in < 0 || d_in >= D_in) continue;

            for (int kh = 0; kh < kH; ++kh) {
                int h_in_nom = h_out + padding - kh;
                if (h_in_nom % stride != 0) continue;
                int h_in = h_in_nom / stride;
                if (h_in < 0 || h_in >= H_in) continue;

                for (int kw = 0; kw < kW; ++kw) {
                    int w_in_nom = w_out + padding - kw;
                    if (w_in_nom % stride != 0) continue;
                    int w_in = w_in_nom / stride;
                    if (w_in < 0 || w_in >= W_in) continue;

                    const int input_idx =
                        (((n * C_in + c_in) * D_in + d_in) * H_in + h_in) * W_in + w_in;

                    // Flip kernel indices to match PyTorch's conv_transpose3d
                    const int weight_idx =
                        ((((c_in * C_out + c_out) * kD + (kD - 1 - kd)) * kH
                          + (kH - 1 - kh)) * kW) + (kW - 1 - kw);

                    val += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    output[tid] = val;
}

////////////////////////////////////////////////////////////////////////////////
// 2. Affine BatchNorm  (y = x * scale[c] + shift[c])
////////////////////////////////////////////////////////////////////////////////
__global__ void batch_norm_affine_kernel(
        const float *__restrict__ input,   // [N, C, D, H, W]
        const float *__restrict__ scale,   // [C]
        const float *__restrict__ shift,   // [C]
        float *__restrict__ output,
        const int N, const int C,
        const int D, const int H, const int W)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * C * D * H * W;
    if (tid >= total) return;

    int w =  tid % W;
    int h = (tid / W) % H;
    int d = (tid / (W * H)) % D;
    int c = (tid / (W * H * D)) % C;
    int n =  tid / (W * H * D * C);

    const int idx = (((n * C + c) * D + d) * H + h) * W + w;
    output[idx] = input[idx] * scale[c] + shift[c];
}

////////////////////////////////////////////////////////////////////////////////
// 3a. Accumulate spatial mean per (n,c)
////////////////////////////////////////////////////////////////////////////////
__global__ void spatial_mean_accum_kernel(
        const float *__restrict__ input, // [N,C,D,H,W]
        float *__restrict__ mean,        // [N,C] (initialized to 0)
        const int N, const int C,
        const int D, const int H, const int W)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * C * D * H * W;
    if (tid >= total) return;

    int w =  tid % W;
    int h = (tid / W) % H;
    int d = (tid / (W * H)) % D;
    int c = (tid / (W * H * D)) % C;
    int n =  tid / (W * H * D * C);

    const int idx = (((n * C + c) * D + d) * H + h) * W + w;
    atomicAdd(&mean[n * C + c], input[idx]);
}

////////////////////////////////////////////////////////////////////////////////
// 3b. Subtract mean
////////////////////////////////////////////////////////////////////////////////
__global__ void mean_subtract_kernel(
        float *__restrict__ data,        // [N,C,D,H,W]
        const float *__restrict__ mean,  // [N,C] (already divided by elem_cnt)
        const int N, const int C,
        const int D, const int H, const int W)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * C * D * H * W;
    if (tid >= total) return;

    int w =  tid % W;
    int h = (tid / W) % H;
    int d = (tid / (W * H)) % D;
    int c = (tid / (W * H * D)) % C;
    int n =  tid / (W * H * D * C);

    const int idx = (((n * C + c) * D + d) * H + h) * W + w;
    data[idx] -= mean[n * C + c];
}

////////////////////////////////////////////////////////////////////////////////
// Host wrappers
////////////////////////////////////////////////////////////////////////////////
torch::Tensor conv_transpose3d_cuda(torch::Tensor input,
                                    torch::Tensor weight,
                                    c10::optional<torch::Tensor> bias_opt,
                                    int stride,
                                    int padding)
{
    const int N = input.size(0);
    const int C_in = input.size(1);
    const int D_in = input.size(2);
    const int H_in = input.size(3);
    const int W_in = input.size(4);

    const int C_out = weight.size(1);
    const int kD = weight.size(2);
    const int kH = weight.size(3);
    const int kW = weight.size(4);

    const int D_out = (D_in - 1) * stride - 2 * padding + kD;
    const int H_out = (H_in - 1) * stride - 2 * padding + kH;
    const int W_out = (W_in - 1) * stride - 2 * padding + kW;

    auto output = torch::empty({N, C_out, D_out, H_out, W_out},
                               input.options());

    const int threads = 256;
    const int blocks  = div_up((int)output.numel(), threads);

    const float* bias_ptr = nullptr;
    if (bias_opt.has_value())
        bias_ptr = bias_opt.value().data_ptr<float>();

    conv_transpose3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        N, C_in,
        D_in, H_in, W_in,
        C_out,
        kD, kH, kW,
        stride, padding,
        D_out, H_out, W_out);

    return output;
}

torch::Tensor batch_norm_affine_cuda(torch::Tensor input,
                                     torch::Tensor scale,
                                     torch::Tensor shift)
{
    const int N = input.size(0);
    const int C = input.size(1);
    const int D = input.size(2);
    const int H = input.size(3);
    const int W = input.size(4);

    auto output = torch::empty_like(input);

    const int threads = 256;
    const int blocks  = div_up((int)input.numel(), threads);

    batch_norm_affine_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        scale.data_ptr<float>(),
        shift.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D, H, W);

    return output;
}

torch::Tensor spatial_mean_subtract_cuda(torch::Tensor input)
{
    const int N = input.size(0);
    const int C = input.size(1);
    const int D = input.size(2);
    const int H = input.size(3);
    const int W = input.size(4);

    auto mean = torch::zeros({N, C}, input.options());

    // 3a. accumulate
    {
        const int threads = 256;
        const int blocks  = div_up((int)input.numel(), threads);
        spatial_mean_accum_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(),
            mean.data_ptr<float>(),
            N, C, D, H, W);
    }

    mean /= static_cast<float>(D * H * W);

    // 3b. subtract
    {
        const int threads = 256;
        const int blocks  = div_up((int)input.numel(), threads);
        mean_subtract_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(),
            mean.data_ptr<float>(),
            N, C, D, H, W);
    }
    return input;
}
"""

cpp_src = """
torch::Tensor conv_transpose3d_cuda(torch::Tensor input,
                                    torch::Tensor weight,
                                    c10::optional<torch::Tensor> bias,
                                    int stride,
                                    int padding);

torch::Tensor batch_norm_affine_cuda(torch::Tensor input,
                                     torch::Tensor scale,
                                     torch::Tensor shift);

torch::Tensor spatial_mean_subtract_cuda(torch::Tensor input);
"""

# Compile kernels
kernels = load_inline(
    name="fused_kernels_fixed",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=[
        "conv_transpose3d_cuda",
        "batch_norm_affine_cuda",
        "spatial_mean_subtract_cuda",
    ],
    verbose=False,
)

# ---------------------------------------------------------------------
# Optimised PyTorch module
# ---------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    ConvTranspose3d + BatchNorm3d (eval-mode affine) + spatial mean subtraction
    executed by custom CUDA kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(in_channels, out_channels, kernel_size, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

        # BatchNorm parameters (affine=True)
        self.bn_weight = nn.Parameter(torch.ones(out_channels))
        self.bn_bias   = nn.Parameter(torch.zeros(out_channels))
        self.register_buffer("running_mean", torch.zeros(out_channels))
        self.register_buffer("running_var",  torch.ones(out_channels))

        self.stride   = stride
        self.padding  = padding
        self.eps      = eps

        # Initialize conv weights similar to PyTorch default
        import math
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5.0))
        if bias:
            fan_in = in_channels * kernel_size * kernel_size * kernel_size
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # ----- ConvTranspose3d -----
        out = kernels.conv_transpose3d_cuda(
            x,
            self.weight.contiguous(),
            self.bias if hasattr(self, "bias") and self.bias is not None else None,
            self.stride,
            self.padding,
        )

        # ----- BatchNorm (affine, inference) -----
        invstd = torch.rsqrt(self.running_var + self.eps)
        scale  = self.bn_weight * invstd
        shift  = self.bn_bias - self.running_mean * scale
        out = kernels.batch_norm_affine_cuda(out, scale.contiguous(), shift.contiguous())

        # ----- Spatial mean subtraction -----
        out = kernels.spatial_mean_subtract_cuda(out)
        return out
