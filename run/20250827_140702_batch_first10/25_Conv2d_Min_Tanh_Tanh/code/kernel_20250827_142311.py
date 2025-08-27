import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------------
# CUDA kernel + bindings
# ------------------------------------------------------------------
cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

/*
    Fused kernel that
      1. takes the minimum over the channel dimension
      2. applies tanh twice
*/
__global__ void min_double_tanh_kernel(const float* __restrict__ x,
                                       float* __restrict__ out,
                                       const int N,
                                       const int C,
                                       const int H,
                                       const int W) {
    const int HW = H * W;
    const int stride = HW;           // distance between successive channels
    const int total = N * HW;        // threads cover (n, h, w) tuples

    CUDA_KERNEL_LOOP(idx, total) {
        int n  = idx / HW;
        int hw = idx % HW;
        int h  = hw / W;
        int w  = hw % W;

        // offset for (n, c=0, h, w)
        int base = ((n * C) * H + h) * W + w;
        float vmin = x[base];

        // channel-wise reduction
        for (int c = 1; c < C; ++c) {
            float v = x[base + c * stride];
            vmin = v < vmin ? v : vmin;
        }

        // two consecutive tanh
        vmin = tanhf(vmin);
        vmin = tanhf(vmin);

        // output memory is (N,1,H,W)
        out[((n * H) + h) * W + w] = vmin;
    }
}

at::Tensor min_double_tanh_cuda(at::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must reside on CUDA device");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 supported");
    TORCH_CHECK(x.dim() == 4, "Input must be 4-D NCHW");

    x = x.contiguous();
    const int N = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);

    auto out = torch::empty({N, 1, H, W}, x.options());

    const int threads = 256;
    const int blocks  = (N * H * W + threads - 1) / threads;

    min_double_tanh_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, H, W);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("CUDA kernel failed: %s\n", cudaGetErrorString(err));

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("min_double_tanh_cuda", &min_double_tanh_cuda,
          "Channel-wise min followed by two tanh (CUDA)");
}
"""

cpp_src = "at::Tensor min_double_tanh_cuda(at::Tensor x);"

min_double_tanh = load_inline(
    name="min_double_tanh",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["min_double_tanh_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

# ------------------------------------------------------------------
# Optimised model
# ------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Optimised version of the original model.
    The channel-wise min and two successive tanh operations are fused
    into a single custom CUDA kernel for improved performance.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self._fused = min_double_tanh

    def forward(self, x):
        x = self.conv(x)
        x = self._fused.min_double_tanh_cuda(x)
        return x
