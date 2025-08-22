import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------------
# CUDA source: GroupNorm (2 kernels)
# ------------------------------------------------------------------
source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/* ---------------------------------------------------------------
   Kernel 1 : compute per-(N,G) mean / var  using atomics
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
    int w = tmp % W; (void)w; tmp /= W;
    int h = tmp % H; (void)h; tmp /= H;
    int c = tmp % C;           tmp /= C;
    int n = tmp;

    int g = c / CpG;
    int mv_idx = n * G + g;

    float v = x[idx];
    atomicAdd(&mean[mv_idx], v);
    atomicAdd(&var[mv_idx],  v * v);
}

/* ---------------------------------------------------------------
   Kernel 2 : apply GroupNorm
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
    int w = tmp % W; (void)w; tmp /= W;
    int h = tmp % H; (void)h; tmp /= H;
    int c = tmp % C;           tmp /= C;
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
    Replacement for the original Model – conv_transpose executed via
    torch builtin; GroupNorm via custom CUDA kernels.
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
            self.stride = (stride, stride)
        else:
            self.stride = stride

        # Original padding (used by the preceding Conv2d layer)
        if isinstance(padding, int):
            p_h = p_w = padding
        else:
            p_h, p_w = padding

        # Adjust padding for the transposed convolution so that the
        # spatial size matches the reference implementation.
        self.trans_padding = (kh - 1 - p_h, kw - 1 - p_w)
        self.output_padding = (p_h, p_w)

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
        # Use PyTorch's highly-optimized conv_transpose2d for correct sizing
        y = torch.nn.functional.conv_transpose2d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.trans_padding,
            output_padding=self.output_padding,
        )
        mean, rvar = kernels.groupnorm_stats(y, self.num_groups, self.eps)
        z = kernels.groupnorm_forward(y, self.gamma, self.beta, mean, rvar, self.num_groups)
        return torch.nn.functional.gelu(z)
