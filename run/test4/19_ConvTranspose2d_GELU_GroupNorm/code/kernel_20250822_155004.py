import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------------
# CUDA source: GroupNorm (apply only – statistics done in Python)
# ------------------------------------------------------------------
source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x)  TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/* ---------------------------------------------------------------
   Element-wise GroupNorm application kernel
-----------------------------------------------------------------*/
template <typename scalar_t>
__global__ void groupnorm_apply_kernel(
        const scalar_t* __restrict__ x,
        const scalar_t* __restrict__ mean,   // [N, G]
        const scalar_t* __restrict__ rvar,   // [N, G]
        const scalar_t* __restrict__ gamma,  // [C]
        const scalar_t* __restrict__ beta,   // [C]
        scalar_t* __restrict__ y,
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

    const int g = c / CpG;
    const int mv_idx = n * G + g;

    const scalar_t m  = mean[mv_idx];
    const scalar_t rv = rvar[mv_idx];
    const scalar_t val = (x[idx] - m) * rv;
    y[idx] = val * gamma[c] + beta[c];
}

/* ---------------------------------------------------------------
   Host wrapper
-----------------------------------------------------------------*/
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

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "groupnorm_apply_kernel", ([&] {
        groupnorm_apply_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            x.data_ptr<scalar_t>(),
            mean.data_ptr<scalar_t>(),
            rvar.data_ptr<scalar_t>(),
            gamma.data_ptr<scalar_t>(),
            beta.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            N, C, H, W,
            G
        );
    }));
    return y;
}
"""

cpp_src = """
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

        # For perfect inversion of Conv2d, padding stays the same,
        # output_padding is 0.
        self.trans_padding   = (p_h, p_w)
        self.output_padding  = (0, 0)

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
        y = torch.nn.functional.conv_transpose2d(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.trans_padding,
            output_padding=self.output_padding,
        )

        N, C, H, W = y.shape
        G = self.num_groups
        y_reshaped = y.view(N, G, -1)
        mean = y_reshaped.mean(dim=2, keepdim=False)
        var  = y_reshaped.var(dim=2, unbiased=False)
        rvar = torch.rsqrt(var + self.eps)

        z = kernels.groupnorm_forward(
            y.contiguous(),
            self.gamma,
            self.beta,
            mean.contiguous(),
            rvar.contiguous(),
            G,
        )
        return torch.nn.functional.gelu(z)
