import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------------
# Custom CUDA kernel: fused “double-Mish” (y = mish(mish(x))) activation
# ---------------------------------------------------------------------------
cpp_src = """
torch::Tensor double_mish_cuda(torch::Tensor input);
"""

cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float mish_activate(float x) {
    float sp = logf(1.0f + expf(x));          // softplus(x)
    return x * tanhf(sp);                     // x * tanh(softplus(x))
}

__global__ void double_mish_kernel(const float* __restrict__ in,
                                   float* __restrict__ out,
                                   int64_t numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    float x = in[idx];
    float y = mish_activate(x);   // first Mish
    y = mish_activate(y);         // second Mish
    out[idx] = y;
}

torch::Tensor double_mish_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "double_mish_cuda: input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32,
                "double_mish_cuda: only float32 tensors are supported");

    auto output = torch::empty_like(input);
    int64_t numel = input.numel();

    const int threads = 256;
    const int blocks  = (numel + threads - 1) / threads;

    double_mish_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        numel);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(err));

    return output;
}
"""

# Build the CUDA extension
double_mish = load_inline(
    name       = "double_mish",
    cpp_sources   = cpp_src,
    cuda_sources  = cuda_src,
    functions     = ["double_mish_cuda"],
    verbose       = False,
)

# ---------------------------------------------------------------------------
# Optimised model
# ---------------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Model with a convolution followed by a fused custom CUDA kernel that applies
    Mish twice in a single pass (mish(mish(x))).
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        # Ensure the convolution lives on GPU so that the custom CUDA kernel
        # can consume its output directly without device transfers.
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size).cuda()
        self.double_mish = double_mish

    def forward(self, x):
        x = self.conv(x)
        x = self.double_mish.double_mish_cuda(x)
        return x

# ---------------------------------------------------------------------------
# Helpers expected by benchmark harness
# ---------------------------------------------------------------------------
batch_size   = 32
in_channels  = 32
out_channels = 64
height = width = 128
kernel_size = 3

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device="cuda")]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
