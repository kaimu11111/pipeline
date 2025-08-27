import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float gelu(float x){
    const float kAlpha = 0.7978845608028654f;  // sqrt(2/pi)
    const float kGamma = 0.044715f;
    return 0.5f * x * (1.0f + tanhf(kAlpha * (x + kGamma * x * x * x)));
}

/*
 * Fused kernel:
 *   – LayerNorm over the last (W) dimension
 *   – GELU activation
 *   – Optional scaling
 *
 * The tensor layout is (N, C, D, H, W) ‑ contiguous in memory, so the W
 * dimension is the fastest-changing (innermost) stride.  The kernel launches
 * one thread per (N, C, D, H) coordinate and performs the reduction over W
 * inside the thread.
 */
__global__ void ln_gelu_scale_lastdim_kernel(const float* __restrict__ input,
                                             float* __restrict__ output,
                                             const int C,
                                             const int D,
                                             const int H,
                                             const int W,
                                             const int elems,   // N*C*D*H
                                             const float eps,
                                             const float scaling){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= elems) return;

    /*  Decompose linear index -> (n, c, d, h)  */
    int h = idx % H;
    int t = idx / H;
    int d = t % D;
    t   /= D;
    int c = t % C;
    int n = t / C;

    /*  Base offset for the first element at width w = 0  */
    size_t base_offset = (((((size_t)n * C + c) * D + d) * H + h) * W);

    /* -------- mean -------- */
    float mean = 0.f;
    for(int w = 0; w < W; ++w){
        mean += input[base_offset + w];
    }
    mean /= static_cast<float>(W);

    /* -------- variance -------- */
    float var = 0.f;
    for(int w = 0; w < W; ++w){
        float diff = input[base_offset + w] - mean;
        var += diff * diff;
    }
    var /= static_cast<float>(W);
    float inv_std = rsqrtf(var + eps);

    /* -------- normalize + GELU + scale -------- */
    for(int w = 0; w < W; ++w){
        float v = input[base_offset + w];
        v = (v - mean) * inv_std;   // LayerNorm over W
        v = gelu(v);                // GELU
        v *= scaling;               // scaling factor
        output[base_offset + w] = v;
    }
}

torch::Tensor fused_layernorm_gelu_scale_cuda(torch::Tensor x,
                                              float eps,
                                              float scaling){
    TORCH_CHECK(x.is_cuda(), "input must reside on CUDA");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "only float32 supported");

    x = x.contiguous();
    auto out = torch::empty_like(x);

    const int N = x.size(0);
    const int C = x.size(1);
    const int D = x.size(2);
    const int H = x.size(3);
    const int W = x.size(4);

    const int elems = N * C * D * H;     // threads-worth of work

    const int threads = 256;
    const int blocks  = (elems + threads - 1) / threads;

    ln_gelu_scale_lastdim_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        C, D, H, W,
        elems,
        eps,
        scaling
    );

    return out;
}
"""

cpp_src = r"""
torch::Tensor fused_layernorm_gelu_scale_cuda(torch::Tensor x,
                                              float eps,
                                              float scaling);
"""

fused_ops = load_inline(
    name="fused_layernorm_gelu_scale",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["fused_layernorm_gelu_scale_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model:
    - ConvTranspose3d
    - Fused LayerNorm (over last dim) + GELU + scaling
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, bias=True, eps=1e-5, scaling_factor=1.0):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias
        )
        self.eps = float(eps)
        self.scaling_factor = float(scaling_factor)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_ops.fused_layernorm_gelu_scale_cuda(
            x, self.eps, self.scaling_factor
        )
        return x
