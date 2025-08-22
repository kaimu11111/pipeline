import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------
# CUDA source: all kernels (fixed kernel flip in ConvTranspose2d)
# ---------------------------------------------------------------------
source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cfloat>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// ------------------------------------------------------------
// 1. Conv-Transpose-2D (N, Cin, H, W) -> (N, Cout, Hout, Wout)
//    NOTE: kernel is flipped (kh, kw) to match PyTorch semantics.
// ------------------------------------------------------------
__global__
void conv_transpose2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int Cin, int Hin, int Win,
    int Cout,
    int kH, int kW,
    int strideH, int strideW,
    int padH, int padW,
    int Hout, int Wout)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * Cout * Hout * Wout;
    if (idx >= total) return;

    int w_out = idx % Wout;
    int h_out = (idx / Wout) % Hout;
    int oc    = (idx / (Wout * Hout)) % Cout;
    int n     = idx / (Wout * Hout * Cout);

    float val = 0.0f;

    // Loop over input channels
    for (int ic = 0; ic < Cin; ++ic)
    {
        // Loop over kernel elements
        for (int kh = 0; kh < kH; ++kh)
        {
            int tmp_h = h_out + padH - kh;
            if (tmp_h < 0) continue;
            if (tmp_h % strideH) continue;
            int h_in = tmp_h / strideH;
            if (h_in >= Hin) continue;

            for (int kw = 0; kw < kW; ++kw)
            {
                int tmp_w = w_out + padW - kw;
                if (tmp_w < 0) continue;
                if (tmp_w % strideW) continue;
                int w_in = tmp_w / strideW;
                if (w_in >= Win) continue;

                // Flip kernel to match PyTorch conv_transpose2d semantics
                int kh_flip = kH - 1 - kh;
                int kw_flip = kW - 1 - kw;

                // weight layout: [Cin, Cout, kH, kW]
                int w_idx  = (((ic * Cout + oc) * kH + kh_flip) * kW + kw_flip);
                int in_idx = (((n * Cin + ic) * Hin + h_in) * Win + w_in);
                val += input[in_idx] * weight[w_idx];
            }
        }
    }
    if (bias != nullptr) val += bias[oc];

    int out_idx = (((n * Cout + oc) * Hout + h_out) * Wout + w_out);
    output[out_idx] = val;
}

torch::Tensor conv_transpose2d_cuda(torch::Tensor input,
                                    torch::Tensor weight,
                                    torch::Tensor bias,
                                    int strideH, int strideW,
                                    int padH, int padW)
{
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    if (bias.defined()) CHECK_INPUT(bias);

    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "weight must be float32");
    if (bias.defined())
        TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "bias must be float32");

    int N  = input.size(0);
    int Cin = input.size(1);
    int Hin = input.size(2);
    int Win = input.size(3);

    int Cout = weight.size(1);
    int kH = weight.size(2);
    int kW = weight.size(3);

    int Hout = (Hin - 1) * strideH - 2 * padH + kH;
    int Wout = (Win - 1) * strideW - 2 * padW + kW;

    auto output = torch::empty({N, Cout, Hout, Wout}, input.options());

    int threads = 256;
    int64_t total = (int64_t)N * Cout * Hout * Wout;
    int blocks = (total + threads - 1) / threads;

    conv_transpose2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, Cin, Hin, Win,
        Cout,
        kH, kW,
        strideH, strideW,
        padH, padW,
        Hout, Wout
    );
    return output;
}

// ------------------------------------------------------------
// 2. MaxPool-2D
// ------------------------------------------------------------
__global__
void maxpool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C,
    int Hin, int Win,
    int kernelH, int kernelW,
    int strideH, int strideW,
    int Hout, int Wout)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * Hout * Wout;
    if (idx >= total) return;

    int w_out = idx % Wout;
    int h_out = (idx / Wout) % Hout;
    int c     = (idx / (Wout * Hout)) % C;
    int n     = idx / (Wout * Hout * C);

    int h_start = h_out * strideH;
    int w_start = w_out * strideW;

    float max_val = -FLT_MAX;
    for (int kh = 0; kh < kernelH; ++kh)
    {
        int h_in = h_start + kh;
        if (h_in >= Hin) continue;
        for (int kw = 0; kw < kernelW; ++kw)
        {
            int w_in = w_start + kw;
            if (w_in >= Win) continue;
            int in_idx = (((n * C + c) * Hin + h_in) * Win + w_in);
            max_val = fmaxf(max_val, input[in_idx]);
        }
    }
    int out_idx = (((n * C + c) * Hout + h_out) * Wout + w_out);
    output[out_idx] = max_val;
}

torch::Tensor maxpool2d_cuda(torch::Tensor input,
                             int kernelH, int kernelW,
                             int strideH, int strideW)
{
    CHECK_INPUT(input);
    int N = input.size(0);
    int C = input.size(1);
    int Hin = input.size(2);
    int Win = input.size(3);

    int Hout = (Hin - kernelH) / strideH + 1;
    int Wout = (Win - kernelW) / strideW + 1;

    auto output = torch::empty({N, C, Hout, Wout}, input.options());

    int threads = 256;
    int64_t total = (int64_t)N * C * Hout * Wout;
    int blocks = (total + threads - 1) / threads;

    maxpool2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, Hin, Win,
        kernelH, kernelW,
        strideH, strideW,
        Hout, Wout
    );
    return output;
}

// ------------------------------------------------------------
// 3. HardTanh – clamp
// ------------------------------------------------------------
__global__
void hardtanh_kernel(const float* input, float* output,
                     int64_t size, float min_val, float max_val)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float v = input[idx];
    v = fminf(fmaxf(v, min_val), max_val);
    output[idx] = v;
}

torch::Tensor hardtanh_cuda(torch::Tensor input, float min_val, float max_val)
{
    CHECK_INPUT(input);
    auto output = torch::empty_like(input);
    int64_t size = input.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    hardtanh_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size, min_val, max_val);
    return output;
}

// ------------------------------------------------------------
// 4. Spatial mean over H & W
// ------------------------------------------------------------
__global__
void spatial_mean_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C;
    if (idx >= total) return;

    int c = idx % C;
    int n = idx / C;

    float sum = 0.0f;
    for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w)
        {
            int in_idx = (((n * C + c) * H + h) * W + w);
            sum += input[in_idx];
        }
    sum /= (H * W);
    output[idx] = sum;
}

torch::Tensor spatial_mean_cuda(torch::Tensor input)
{
    CHECK_INPUT(input);
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    auto output = torch::empty({N, C, 1, 1}, input.options());

    int threads = 256;
    int64_t total = (int64_t)N * C;
    int blocks = (total + threads - 1) / threads;
    spatial_mean_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W
    );
    return output;
}

// ------------------------------------------------------------
// 5. Tanh – element-wise
// ------------------------------------------------------------
__global__
void tanh_kernel(const float* input, float* output, int64_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    output[idx] = tanhf(input[idx]);
}

torch::Tensor tanh_cuda(torch::Tensor input)
{
    CHECK_INPUT(input);
    auto output = torch::empty_like(input);
    int64_t size = input.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    tanh_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );
    return output;
}
"""

# ---------------------------------------------------------------------
# C++ prototypes for exposed functions
# ---------------------------------------------------------------------
cpp_src = """
torch::Tensor conv_transpose2d_cuda(torch::Tensor input,
                                    torch::Tensor weight,
                                    torch::Tensor bias,
                                    int strideH, int strideW,
                                    int padH, int padW);
torch::Tensor maxpool2d_cuda(torch::Tensor input,
                             int kernelH, int kernelW,
                             int strideH, int strideW);
torch::Tensor hardtanh_cuda(torch::Tensor input, float min_val, float max_val);
torch::Tensor spatial_mean_cuda(torch::Tensor input);
torch::Tensor tanh_cuda(torch::Tensor input);
"""

# ---------------------------------------------------------------------
# Build & load kernels
# ---------------------------------------------------------------------
kernels = load_inline(
    name="custom_ops_modelnew_fix",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=[
        "conv_transpose2d_cuda",
        "maxpool2d_cuda",
        "hardtanh_cuda",
        "spatial_mean_cuda",
        "tanh_cuda"
    ],
    verbose=False
)

# ---------------------------------------------------------------------
# ModelNew definition
# ---------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Optimised model replacing select PyTorch ops with custom CUDA kernels.
    """
    def __init__(self,
                 in_channels, out_channels,
                 kernel_size, stride, padding,
                 maxpool_kernel_size, maxpool_stride,
                 hardtanh_min, hardtanh_max):
        super().__init__()

        # Weight shape matches PyTorch's ConvTranspose2d: (in_channels, out_channels, kH, kW)
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, kernel_size, kernel_size, device='cuda', dtype=torch.float32)
        )
        self.bias = nn.Parameter(torch.randn(out_channels, device='cuda', dtype=torch.float32))

        # --- hyper-parameters ---
        if isinstance(stride, int):
            self.stride_h = self.stride_w = stride
        else:
            self.stride_h, self.stride_w = stride

        if isinstance(padding, int):
            self.pad_h = self.pad_w = padding
        else:
            self.pad_h, self.pad_w = padding

        if isinstance(maxpool_kernel_size, int):
            self.pool_kernel_h = self.pool_kernel_w = maxpool_kernel_size
        else:
            self.pool_kernel_h, self.pool_kernel_w = maxpool_kernel_size

        if isinstance(maxpool_stride, int):
            self.pool_stride_h = self.pool_stride_w = maxpool_stride
        else:
            self.pool_stride_h, self.pool_stride_w = maxpool_stride

        self.ht_min = float(hardtanh_min)
        self.ht_max = float(hardtanh_max)

    def forward(self, x):
        # ConvTranspose2d
        x = kernels.conv_transpose2d_cuda(
            x, self.weight, self.bias,
            self.stride_h, self.stride_w,
            self.pad_h, self.pad_w
        )
        # MaxPool2d
        x = kernels.maxpool2d_cuda(
            x,
            self.pool_kernel_h, self.pool_kernel_w,
            self.pool_stride_h, self.pool_stride_w
        )
        # HardTanh
        x = kernels.hardtanh_cuda(x, self.ht_min, self.ht_max)
        # Spatial mean
        x = kernels.spatial_mean_cuda(x)
        # Tanh activation
        x = kernels.tanh_cuda(x)
        return x
