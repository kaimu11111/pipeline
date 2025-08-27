import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------------
# CUDA kernels and host wrapper (no PYBIND block; load_inline will generate it)
# ---------------------------------------------------------------------------
source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

////////////////////////////////////////////////////////////////////////////////
// Kernel 1: accumulate sums per (n, c)
////////////////////////////////////////////////////////////////////////////////
__global__ void compute_sums_kernel(const float* __restrict__ x,
                                    float* __restrict__ sums,
                                    const long total_elements,
                                    const int C,
                                    const int D,
                                    const int H,
                                    const int W) {
    const long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    long tmp = idx;
    const int w = tmp % W; (void)w; tmp /= W;
    const int h = tmp % H; (void)h; tmp /= H;
    const int d = tmp % D; (void)d; tmp /= D;
    const int c = tmp % C;
    const int n = tmp / C;
    const int nc_index = n * C + c;

    atomicAdd(&sums[nc_index], x[idx]);
}

////////////////////////////////////////////////////////////////////////////////
// Kernel 2: subtract mean from each element
////////////////////////////////////////////////////////////////////////////////
__global__ void subtract_mean_kernel(const float* __restrict__ x,
                                     const float* __restrict__ sums,
                                     float* __restrict__ out,
                                     const long total_elements,
                                     const int C,
                                     const int D,
                                     const int H,
                                     const int W) {
    const long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    long tmp = idx;
    tmp /= W;
    tmp /= H;
    const int d = tmp % D; (void)d;
    tmp /= D;
    const int c = tmp % C;
    const int n = tmp / C;
    const int nc_index = n * C + c;

    const float mean = sums[nc_index] / static_cast<float>(D * H * W);
    out[idx] = x[idx] - mean;
}

////////////////////////////////////////////////////////////////////////////////
// Host function callable from Python
////////////////////////////////////////////////////////////////////////////////
torch::Tensor subtract_mean_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 5, "Expected input of shape (N, C, D, H, W)");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Only float32 tensors are supported");

    if (!x.is_contiguous()) {
        x = x.contiguous();
    }

    const int N = x.size(0);
    const int C = x.size(1);
    const int D = x.size(2);
    const int H = x.size(3);
    const int W = x.size(4);

    auto sums = torch::zeros({N, C}, x.options());
    auto out  = torch::empty_like(x);

    const long total_elements = x.numel();
    constexpr int threads = 256;
    const int blocks = static_cast<int>((total_elements + threads - 1) / threads);

    compute_sums_kernel<<<blocks, threads>>>(x.data_ptr<float>(),
                                             sums.data_ptr<float>(),
                                             total_elements, C, D, H, W);

    subtract_mean_kernel<<<blocks, threads>>>(x.data_ptr<float>(),
                                              sums.data_ptr<float>(),
                                              out.data_ptr<float>(),
                                              total_elements, C, D, H, W);
    return out;
}
"""

cpp_src = "torch::Tensor subtract_mean_cuda(torch::Tensor x);"

subtract_mean = load_inline(
    name="subtract_mean",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["subtract_mean_cuda"],
    verbose=False,
)

# ---------------------------------------------------------------------------
# Optimised model
# ---------------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    ConvTranspose3d + BatchNorm3d followed by a custom CUDA kernel that
    subtracts the spatial mean (over D, H, W) for every (N, C) slice.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias
        )
        self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        x = subtract_mean.subtract_mean_cuda(x)
        return x
