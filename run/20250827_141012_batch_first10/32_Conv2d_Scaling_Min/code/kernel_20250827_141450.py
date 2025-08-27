import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------------
# CUDA kernel + launcher (no explicit PYBIND block; load_inline adds it)
# ------------------------------------------------------------------
source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

namespace {

__global__ void scale_min_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 const float  scale_factor,
                                 const int    N,
                                 const int    C,
                                 const int    H,
                                 const int    W) {
    const int idx          = blockIdx.x * blockDim.x + threadIdx.x;
    const int spatial_size = H * W;
    const int output_size  = N * spatial_size;
    if (idx >= output_size) return;

    const int n  = idx / spatial_size;
    const int hw = idx % spatial_size;
    const int h  = hw / W;
    const int w  = hw % W;

    float min_val = FLT_MAX;
    for (int c = 0; c < C; ++c) {
        const int offset = ((n * C + c) * H + h) * W + w;
        float v = input[offset] * scale_factor;
        if (v < min_val) min_val = v;
    }
    output[idx] = min_val;
}

} // anonymous namespace

torch::Tensor scale_min_cuda(torch::Tensor input, const double scale_factor_double) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32,
                "only float32 tensors are supported");

    auto input_contig = input.contiguous();
    const int64_t N = input_contig.size(0);
    const int64_t C = input_contig.size(1);
    const int64_t H = input_contig.size(2);
    const int64_t W = input_contig.size(3);

    auto output = torch::empty({N, 1, H, W}, input_contig.options());

    const int threads = 256;
    const int blocks  = (N * H * W + threads - 1) / threads;

    scale_min_kernel<<<blocks, threads>>>(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<float>(scale_factor_double),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}
"""

# ------------------------------------------------------------------
# Function prototype(s) for automatic Python bindings
# ------------------------------------------------------------------
cpp_src = """
torch::Tensor scale_min_cuda(torch::Tensor input, const double scale_factor_double);
"""

# ------------------------------------------------------------------
# Build / load the CUDA extension
# ------------------------------------------------------------------
scale_min = load_inline(
    name="scale_min",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["scale_min_cuda"],
    verbose=False,
)

# ------------------------------------------------------------------
# Optimised model that calls the fused CUDA kernel
# ------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super().__init__()
        # Remove padding to match the reference model’s behaviour
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0)
        self.scale_factor = float(scale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = scale_min.scale_min_cuda(x, self.scale_factor)
        return x.contiguous()
