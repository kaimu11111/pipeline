import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Inline CUDA/C++ code for fused maxpool, scale multiplication, and clamp
source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void fused_maxpool_scale_clamp_kernel(
    const float* __restrict__ x,
    const float* __restrict__ scale,
    float* __restrict__ out,
    int B, int C, int H, int W, 
    int outH, int outW,
    int pool_size,
    float clamp_min,
    float clamp_max
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B*C*outH*outW;
    if (tid >= total) return;

    int b = tid / (C * outH * outW);
    int r = tid % (C * outH * outW);
    int c = r / (outH * outW);
    r = r % (outH * outW);
    int oh = r / outW;
    int ow = r % outW;

    float max_val = -FLT_MAX;
    // read the scale just once for this channel
    float s = scale.size(0) == C ? scale[c] : scale[c * scale.size(1) * scale.size(2)];

    int h_start = oh * pool_size;
    int w_start = ow * pool_size;
    for (int i = 0; i < pool_size; i++) {
        for (int j = 0; j < pool_size; j++) {
            int h_in = h_start + i;
            int w_in = w_start + j;
            float val = x[((b * C + c) * H + h_in) * W + w_in] * s;
            max_val = fmaxf(max_val, val);
        }
    }
    max_val = fminf(fmaxf(max_val, clamp_min), clamp_max);
    out[((b * C + c) * outH + oh) * outW + ow] = max_val;
}

torch::Tensor fused_maxpool_scale_clamp_cuda(
    torch::Tensor x, 
    torch::Tensor scale,
    int pool_size, 
    float clamp_min, 
    float clamp_max
) {
    TORCH_CHECK(x.dim() == 4, "x must be 4D (B, C, H, W)");
    TORCH_CHECK(scale.dim() == 3 || scale.dim() == 1, "scale must be shape (C, 1, 1) or (C)");
    auto B = x.size(0);
    auto C = x.size(1);
    auto H = x.size(2);
    auto W = x.size(3);

    int outH = H / pool_size;
    int outW = W / pool_size;
    auto out = torch::empty({B, C, outH, outW}, x.options());

    int total = B * C * outH * outW;
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;

    fused_maxpool_scale_clamp_kernel<<<gridSize, blockSize>>>(
        x.data_ptr<float>(),
        scale.data_ptr<float>(),
        out.data_ptr<float>(),
        B, C, H, W,
        outH, outW,
        pool_size,
        clamp_min,
        clamp_max
    );

    return out;
}
"""

cpp_source = r"""
torch::Tensor fused_maxpool_scale_clamp_cuda(
    torch::Tensor x, 
    torch::Tensor scale,
    int pool_size, 
    float clamp_min, 
    float clamp_max
);
"""

# Load and build the fused CUDA extension
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=cpp_source,
    cuda_sources=source,
    functions=["fused_maxpool_scale_clamp_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    """
    Optimized model that fuses scaling, max pooling, and clamping into a single custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.maxpool_kernel_size = maxpool_kernel_size
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        x = self.conv(x)
        x = self.group_norm(x)
        # Fused: (x * scale) -> maxpool -> clamp
        x = fused_ops.fused_maxpool_scale_clamp_cuda(
            x, self.scale, 
            self.maxpool_kernel_size, 
            self.clamp_min,
            self.clamp_max
        )
        return x
