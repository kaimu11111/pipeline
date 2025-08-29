# <complete ModelNew code>
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# --------------------------------------------------------------------
# CUDA kernel v3 : vectorised + unrolled fuse(min, bias, scale)
# --------------------------------------------------------------------
source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#endif

constexpr int VEC_WIDTH = 4;            // float4 = 4 * 4  = 16-byte
using float4_t = float4;                // helper alias

// ------------------------------------------------------------------
// Vectorised kernel
// grid  = (channels, batch)          -> 1 block = (n,c) pair
// block = 128 threads (config in host)
// every thread processes several float4 packs (4 elements) so total
// processed elements per iter = blockDim.x * VEC_WIDTH
// ------------------------------------------------------------------
__global__ void fuse_kernel_v3(const float* __restrict__ x,
                               const float* __restrict__ bias,
                               float* __restrict__ out,
                               const int64_t spatial_size,
                               const int64_t channels,
                               const float   constant_val,
                               const float   scale) {
    const int c = blockIdx.x;   // channel
    const int n = blockIdx.y;   // batch index
    if (c >= channels) return;

    // base pointer for this (n,c) slice
    const int64_t base = ( (int64_t)n * channels + c ) * spatial_size;

    const float bias_val = bias[c];

    // float4 vectorised ptrs – only valid when base is 16-byte aligned
    const float*  src_ptr = x   + base;
    float*        dst_ptr = out + base;

    const int64_t vec_elems   = spatial_size / VEC_WIDTH;   // full float4 packs
    const int64_t tail_start  = vec_elems * VEC_WIDTH;      // scalar tail offset

    // ------------- Vectorised part (float4) -------------
    for (int64_t i = threadIdx.x; i < vec_elems; i += blockDim.x) {
        const int64_t offset = i * VEC_WIDTH;

        // Aligned load via reinterpret_cast – safe because originating
        // PyTorch tensor is 4-byte aligned and we only use when offset
        // is multiple of 4 elements.
        float4_t v4 = reinterpret_cast<const float4_t*>(src_ptr)[i];

        // operate element-wise
        float4_t r4;
        r4.x = fminf(v4.x, constant_val);
        r4.y = fminf(v4.y, constant_val);
        r4.z = fminf(v4.z, constant_val);
        r4.w = fminf(v4.w, constant_val);

        r4.x = (r4.x + bias_val) * scale;
        r4.y = (r4.y + bias_val) * scale;
        r4.z = (r4.z + bias_val) * scale;
        r4.w = (r4.w + bias_val) * scale;

        reinterpret_cast<float4_t*>(dst_ptr)[i] = r4;
    }

    // ------------- Scalar tail (if spatial_size not divisible by 4) -------------
    for (int64_t s = tail_start + threadIdx.x; s < spatial_size; s += blockDim.x) {
        float v = src_ptr[s];
        v = v < constant_val ? v : constant_val;
        v = (v + bias_val) * scale;
        dst_ptr[s] = v;
    }
}

// ------------------------------------------------------------------
// Host wrapper
// ------------------------------------------------------------------
torch::Tensor fuse_min_bias_scale_cuda(torch::Tensor x,
                                       torch::Tensor bias,
                                       float constant_val,
                                       float scale) {
    CHECK_CUDA(x);
    CHECK_CUDA(bias);
    TORCH_CHECK(x.dtype()   == torch::kFloat32, "x must be float32");
    TORCH_CHECK(bias.dtype()== torch::kFloat32, "bias must be float32");

    x    = x.contiguous();
    bias = bias.contiguous().view({-1});   // (C)

    auto out = torch::empty_like(x);

    const int64_t N       = x.size(0);
    const int64_t C       = x.size(1);
    const int64_t spatial = x.size(2) * x.size(3);  // H*W

    // launch configuration
    const dim3 block(128);
    const dim3 grid(C, N);

    fuse_kernel_v3<<<grid, block>>>(
        x.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        spatial,
        C,
        constant_val,
        scale);

    return out;
}
"""

cpp_src = """
torch::Tensor fuse_min_bias_scale_cuda(torch::Tensor x,
                                       torch::Tensor bias,
                                       float constant_val,
                                       float scale);
"""

# Compile / load the inline extension (single compilation unit)
_fused_ops = load_inline(name="fuse_min_bias_scale_v3",
                         cpp_sources=cpp_src,
                         cuda_sources=source,
                         functions=["fuse_min_bias_scale_cuda"],
                         verbose=False)

# --------------------
# Optimised torch model
# --------------------
class ModelNew(nn.Module):
    """
    PyTorch module that runs a convolution and
    fuses min(+constant) + bias add + scalar multiply
    via a custom hand-written CUDA kernel (v3).
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size, constant_value,
                 bias_shape, scaling_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.register_buffer("constant_value",
                             torch.tensor(float(constant_value)))
        self.register_buffer("scaling_factor",
                             torch.tensor(float(scaling_factor)))
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv(x)
        x = _fused_ops.fuse_min_bias_scale_cuda(
            x,
            self.bias.view(-1),
            float(self.constant_value),
            float(self.scaling_factor)
        )
        return x
