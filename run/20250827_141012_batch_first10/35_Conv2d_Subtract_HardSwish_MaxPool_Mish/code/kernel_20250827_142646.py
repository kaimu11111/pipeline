import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------------
# Hand–written CUDA kernels (device + host wrappers, NO pybind here)
# ------------------------------------------------------------------
source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// ------------------------------------------------------
// subtract + HardSwish kernel
// y = hardswish(x - subtract_val)
// ------------------------------------------------------
__global__ void sub_hswish_kernel(const float* __restrict__ x,
                                  float* __restrict__ y,
                                  const float subtract_val,
                                  const int64_t numel) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float v = x[idx] - subtract_val;
        float h = fminf(fmaxf(v + 3.0f, 0.0f), 6.0f);   // relu6(v+3)
        y[idx] = v * (h / 6.0f);
    }
}

// ------------------------------------------------------
// Mish kernel : y = x * tanh(softplus(x))
// ------------------------------------------------------
__global__ void mish_kernel(const float* __restrict__ x,
                            float* __restrict__ y,
                            const int64_t numel) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float v  = x[idx];
        float sp = log1pf(expf(v));          // softplus
        y[idx]   = v * tanhf(sp);
    }
}

// ------------------------------------------------------
// C++ / ATen wrappers (visible to Python)
// ------------------------------------------------------
torch::Tensor subtract_hardswish_cuda(torch::Tensor x, const float subtract_val) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat, "Only float32 supported");

    auto y = torch::empty_like(x);

    const int64_t numel = x.numel();
    const int  block   = 256;
    const int  grid    = (numel + block - 1) / block;

    sub_hswish_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        subtract_val,
        numel
    );

    return y;
}

torch::Tensor mish_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat, "Only float32 supported");

    auto y = torch::empty_like(x);

    const int64_t numel = x.numel();
    const int  block   = 256;
    const int  grid    = (numel + block - 1) / block;

    mish_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        numel
    );

    return y;
}
"""

# ------------------------------------------------------------------
# Function prototypes exposed to Python
# ------------------------------------------------------------------
cpp_src = """
torch::Tensor subtract_hardswish_cuda(torch::Tensor x, float subtract_val);
torch::Tensor mish_cuda(torch::Tensor x);
"""

# ------------------------------------------------------------------
# Build / load the fused activation extension
# ------------------------------------------------------------------
fused_act = load_inline(
    name="fused_act",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["subtract_hardswish_cuda", "mish_cuda"],
    verbose=False,
)

# ------------------------------------------------------------------
# Optimised model
# ------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Optimised model using custom CUDA kernels for
    1) subtract + HardSwish
    2) Mish
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 subtract_value, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value = float(subtract_value)
        self.pool = nn.MaxPool2d(pool_kernel_size)

    def forward(self, x):
        x = self.conv(x)
        # Custom fused subtract + HardSwish
        x = fused_act.subtract_hardswish_cuda(x, self.subtract_value)
        x = self.pool(x)
        # Custom Mish
        x = fused_act.mish_cuda(x)
        return x
