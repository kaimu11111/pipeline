import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------------
# CUDA source (kernel + host wrapper, NO PYBIND11_MODULE here!)
# ---------------------------------------------------------------------------
source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Kernel: one block per sample (batch element), thread-level strided reduction
// ---------------------------------------------------------------------------
__global__ void batch_mean_kernel(const float* __restrict__ x,
                                  float* __restrict__ out,
                                  const int inner_size)
{
    const int n = blockIdx.x;          // sample (batch) index
    float thread_sum = 0.f;

    // Strided read over this sample
    for (int idx = threadIdx.x; idx < inner_size; idx += blockDim.x)
        thread_sum += x[n * inner_size + idx];

    // Block-level reduction
    extern __shared__ float shm[];
    shm[threadIdx.x] = thread_sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            shm[threadIdx.x] += shm[threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        out[n] = shm[0] / static_cast<float>(inner_size);
}

// ---------------------------------------------------------------------------
// Host wrapper
// ---------------------------------------------------------------------------
torch::Tensor batch_mean_cuda(torch::Tensor x)
{
    TORCH_CHECK(x.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32,
                "only float32 tensors are supported");

    const int64_t N = x.size(0);
    const int64_t inner_size = x.numel() / N;

    auto out = torch::empty({N}, x.options().dtype(torch::kFloat32));

    const int threads = 256;
    const dim3 blocks(N);
    const size_t smem = threads * sizeof(float);

    batch_mean_kernel<<<blocks, threads, smem>>>(x.data_ptr<float>(),
                                                 out.data_ptr<float>(),
                                                 static_cast<int>(inner_size));
    return out;
}
"""

# ---------------------------------------------------------------------------
# C++ prototypes exposed to Python
# ---------------------------------------------------------------------------
cpp_src = "torch::Tensor batch_mean_cuda(torch::Tensor x);"

# ---------------------------------------------------------------------------
# Build / load extension
# ---------------------------------------------------------------------------
batch_mean_ext = load_inline(
    name="batch_mean_ext",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["batch_mean_cuda"],
    verbose=False,
)

# ---------------------------------------------------------------------------
# PyTorch module utilising the custom kernel
# ---------------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    3D Conv → GroupNorm → per-sample mean (custom CUDA kernel)
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure sub-modules are on the same device as the input
        dev = x.device
        if self.conv.weight.device != dev:
            self.conv.to(dev)
            self.group_norm.to(dev)

        x = self.conv(x)
        x = self.group_norm(x)
        x = batch_mean_ext.batch_mean_cuda(x)
        return x
