import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------------
# CUDA source: transposed conv2d (+padding)  and GroupNorm (2-kernel)
# ------------------------------------------------------------------
source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/* ---------------------------------------------------------------
   Kernel 1 : 2-D Transposed Convolution (with padding, no activation)
-----------------------------------------------------------------*/
__global__ void conv_transpose2d_kernel(
        const float* __restrict__ inp,      // [N, C_in, H_in, W_in]
        const float* __restrict__ weight,   // [C_in, C_out, kH, kW]
        const float* __restrict__ bias,     // [C_out] (can be null)
        float* __restrict__ out,            // [N, C_out, H_out, W_out]
        int N, int C_in, int C_out,
        int H_in, int W_in,
        int H_out, int W_out,
        int kH, int kW,
        int stride_h, int stride_w,
        int pad_h, int pad_w)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * C_out * H_out * W_out;
    if (idx >= total) return;

    int tmp = idx;
    const int ow = tmp % W_out;  tmp /= W_out;
    const int oh = tmp % H_out;  tmp /= H_out;
    const int oc = tmp % C_out;  tmp /= C_out;
    const int n  = tmp;

    float val = 0.f;

    for (int ic = 0; ic < C_in; ++ic) {
        for (int kh = 0; kh < kH; ++kh) {
            int ih_nom = oh + pad_h - kh;
            if (ih_nom < 0 || ih_nom % stride_h != 0) continue;
            int ih = ih_nom / stride_h;
            if (ih < 0 || ih >= H_in) continue;

            for (int kw = 0; kw < kW; ++kw) {
                int iw_nom = ow + pad_w - kw;
                if (iw_nom < 0 || iw_nom % stride_w != 0) continue;
                int iw = iw_nom / stride_w;
                if (iw < 0 || iw >= W_in) continue;

                const int inp_off = (((n * C_in + ic) * H_in + ih) * W_in + iw);
                const int w_off   = (((ic * C_out + oc) * kH + kh) * kW + kw);
                val += inp[inp_off] * weight[w_off];
            }
        }
    }
    if (bias != nullptr) val += bias[oc];
    out[idx] = val;
}

/* ---------------------------------------------------------------
   Kernel 2 : compute per-(N,G) mean / var  using atomics
-----------------------------------------------------------------*/
__global__ void groupnorm_mean_var_kernel(
        const float* __restrict__ x,
        float* __restrict__ mean,   // [N, G]
        float* __restrict__ var,    // [N, G]
        int N, int C, int H, int W,
        int G)
{
    const int CpG = C / G;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * C * H * W;
    if (idx >= total) return;

    int tmp = idx;
    int w = tmp % W; tmp /= W;
    int h = tmp % H; tmp /= H;
    int c = tmp % C; tmp /= C;
    int n = tmp;

    int g = c / CpG;
    int mv_idx = n * G + g;

    float v = x[idx];
    atomicAdd(&mean[mv_idx], v);
    atomicAdd(&var[mv_idx],  v * v);
}

/* ---------------------------------------------------------------
   Kernel 3 : apply GroupNorm
-----------------------------------------------------------------*/
__global__ void groupnorm_apply_kernel(
        const float* __restrict__ x,
        const float* __restrict__ mean,   // [N, G]
        const float* __restrict__ rvar,   // reciprocal std-dev [N, G]
        const float* __restrict__ gamma,  // [C]
        const float* __restrict__ beta,   // [C]
        float* __restrict__ y,
        int N, int C, int H, int W,
        int G)
{
    const int CpG = C / G;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * C * H * W;
    if (idx >= total) return;

    int tmp = idx;
    int w = tmp % W; tmp /= W;
    int h = tmp % H; tmp /= H;
    int c = tmp % C; tmp /= C;
    int n = tmp;

    int g = c / CpG;
    int mv_idx = n * G + g;

    float m  = mean[mv_idx];
    float rv = rvar[mv_idx];
    float v  = (x[idx] - m) * rv;
    v = v * gamma[c] + beta[c];
    y[idx] = v;
}

/* ---------------------------------------------------------------
   C++ interfaces
-----------------------------------------------------------------*/
torch::Tensor conv_transpose2d_forward(
        torch::Tensor x,
        torch::Tensor w,
        torch::Tensor b,
        int stride_h, int stride_w,
        int pad_h, int pad_w)
{
    CHECK_INPUT(x);
    CHECK_INPUT(w);
    if (b.defined()) CHECK_INPUT(b);

    const int N = x.size(0);
    const int C_in = x.size(1);
    const int H_in = x.size(2);
    const int W_in = x.size(3);

    const int kH = w.size(2);
    const int kW = w.size(3);
    const int C_out = w.size(1);

    const int H_out = (H_in - 1) * stride_h - 2 * pad_h + kH;
    const int W_out = (W_in - 1) * stride_w - 2 * pad_w + kW;

    auto out = torch::empty({N, C_out, H_out, W_out}, x.options());

    const int threads = 256;
    const int total = N * C_out * H_out * W_out;
    const int blocks = (total + threads - 1) / threads;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    conv_transpose2d_kernel<<<blocks, threads, 0, stream>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        b.defined() ? b.data_ptr<float>() : nullptr,
        out.data_ptr<float>(),
        N, C_in, C_out,
        H_in, W_in,
        H_out, W_out,
        kH, kW,
        stride_h, stride_w,
        pad_h, pad_w
    );
    return out;
}

std::tuple<torch::Tensor, torch::Tensor> groupnorm_stats(
        torch::Tensor x,
        int G,
        float eps)
{
    CHECK_INPUT(x);
    const int N = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);

    auto mean = torch::zeros({N, G}, x.options());
    auto var  = torch::zeros({N, G}, x.options());

    const int threads = 256;
    const int total = N * C * H * W;
    const int blocks = (total + threads - 1) / threads;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    groupnorm_mean_var_kernel<<<blocks, threads, 0, stream>>>(
        x.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        N, C, H, W,
        G
    );

    const int M = (C / G) * H * W;
    mean = mean / M;
    var  = var / M - mean * mean;
    auto rvar = (var + eps).rsqrt();
    return {mean, rvar};
}

torch::Tensor groupnorm_forward(
        torch::Tensor x,
        torch::Tensor gamma,
        torch::Tensor beta,
        torch::Tensor mean,
        torch::Tensor rvar,
        int G)
{
    CHECK_INPUT(x);
    CHECK_INPUT(gamma);
    CHECK_INPUT(beta);
    CHECK_INPUT(mean);
    CHECK_INPUT(rvar);

    const int N = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);

    auto y = torch::empty_like(x);

    const int threads = 256;
    const int total = N * C * H * W;
    const int blocks = (total + threads - 1) / threads;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    groupnorm_apply_kernel<<<blocks, threads, 0, stream>>>(
        x.data_ptr<float>(),
        mean.data_ptr<float>(),
        rvar.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        y.data_ptr<float>(),
        N, C, H, W,
        G
    );
    return y;
}
"""

cpp_src = """
torch::Tensor conv_transpose2d_forward(torch::Tensor x, torch::Tensor w, torch::Tensor b,
                                       int stride_h, int stride_w, int pad_h, int pad_w);
std::tuple<torch::Tensor, torch::Tensor> groupnorm_stats(torch::Tensor x, int G, float eps);
torch::Tensor groupnorm_forward(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta,
                                torch::Tensor mean, torch::Tensor rvar, int G);
"""

# ------------------------------------------------------------------
# Build extension
# ------------------------------------------------------------------
kernels = load_inline(
    name="model_kernels_fix",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=[
        "conv_transpose2d_forward",
        "groupnorm_stats",
        "groupnorm_forward",
    ],
    verbose=False,
)

# ------------------------------------------------------------------
# Python module
# ------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Replacement for the original Model – heavy ops executed via custom CUDA kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, num_groups, eps: float = 1e-5):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels

        if isinstance(kernel_size, int):
            kh = kw = kernel_size
        else:
            kh, kw = kernel_size

        if isinstance(stride, int):
            self.stride_h = self.stride_w = stride
        else:
            self.stride_h, self.stride_w = stride

        if isinstance(padding, int):
            self.pad_h = self.pad_w = padding
        else:
            self.pad_h, self.pad_w = padding

        # Parameters for transposed convolution
        weight = torch.empty(in_channels, out_channels, kh, kw)
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        self.weight = nn.Parameter(weight)

        bias = torch.empty(out_channels)
        fan_in = in_channels * kh * kw
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(bias, -bound, bound)
        self.bias = nn.Parameter(bias)

        # Parameters for GroupNorm
        self.gamma = nn.Parameter(torch.ones(out_channels))
        self.beta  = nn.Parameter(torch.zeros(out_channels))
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        y = kernels.conv_transpose2d_forward(
            x, self.weight, self.bias,
            self.stride_h, self.stride_w,
            self.pad_h, self.pad_w
        )
        mean, rvar = kernels.groupnorm_stats(y, self.num_groups, self.eps)
        z = kernels.groupnorm_forward(y, self.gamma, self.beta, mean, rvar, self.num_groups)
        return torch.nn.functional.gelu(z)
