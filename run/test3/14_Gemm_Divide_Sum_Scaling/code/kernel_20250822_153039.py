import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------
# CUDA source code (kernels + fused_forward implementation ONLY)
# ---------------------------------------------------------------------
source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>   // <-- provides getCurrentCUDAStream
#include <cuda_runtime.h>

/////////////////////////////////////////////////////////////////
// Kernel 1 : column-wise sum  (wsum[col] = sum_h weight[h,col])
/////////////////////////////////////////////////////////////////
__global__ void col_sum_kernel(const float* __restrict__ weight,
                               float* __restrict__ wsum,
                               int hidden_size, int input_size) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= input_size) return;

    float s = 0.f;
    for (int row = 0; row < hidden_size; ++row) {
        s += weight[row * input_size + col];
    }
    wsum[col] = s;
}

/////////////////////////////////////////////////////////////////
// Kernel 2 : per-batch dot product with reduction in shared mem
/////////////////////////////////////////////////////////////////
template <unsigned int BLOCK_SIZE>
__global__ void batch_dot_kernel(const float* __restrict__ x,
                                 const float* __restrict__ wsum,
                                 float* __restrict__ out,
                                 float scale,
                                 int batch_size, int input_size) {
    extern __shared__ float shm[];                   // BLOCK_SIZE floats
    int b = blockIdx.x;                              // one block per batch row

    float local = 0.f;
    // Strided loop over input dimension
    for (int idx = threadIdx.x; idx < input_size; idx += BLOCK_SIZE) {
        local += x[b * input_size + idx] * wsum[idx];
    }
    shm[threadIdx.x] = local;
    __syncthreads();

    // Reduction (BLOCK_SIZE assumed power of 2)
    for (unsigned int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) shm[threadIdx.x] += shm[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out[b] = shm[0] * scale;
    }
}

/////////////////////////////////////////////////////////////////
// Host wrapper
/////////////////////////////////////////////////////////////////
std::tuple<torch::Tensor, torch::Tensor> fused_forward_cuda(
        torch::Tensor x,
        torch::Tensor weight,
        float   scale) {

    const int batch_size  = x.size(0);
    const int input_size  = x.size(1);
    const int hidden_size = weight.size(0);

    // Allocate intermediate and output tensors
    auto wsum = torch::empty({input_size}, x.options());
    auto out  = torch::empty({batch_size},  x.options());

    // Launch kernel 1 : column sums
    const int COL_BLOCK = 256;
    const dim3 grid1((input_size + COL_BLOCK - 1) / COL_BLOCK);
    const dim3 block1(COL_BLOCK);
    col_sum_kernel<<<grid1, block1, 0, at::cuda::getCurrentCUDAStream()>>>(
        weight.data_ptr<float>(),
        wsum.data_ptr<float>(),
        hidden_size, input_size);

    // Launch kernel 2 : batch dot products
    const int DOT_BLOCK = 256;
    const dim3 grid2(batch_size);
    const dim3 block2(DOT_BLOCK);
    const size_t shm_bytes = DOT_BLOCK * sizeof(float);

    batch_dot_kernel<DOT_BLOCK><<<grid2, block2, shm_bytes, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(),
        wsum.data_ptr<float>(),
        out.data_ptr<float>(),
        scale,
        batch_size, input_size);

    // Return both output tensors (out, wsum) – caller uses out
    return std::make_tuple(out, wsum);
}
"""

# ---------------------------------------------------------------------
# C++ declaration code (NO PYBIND11_MODULE here; load_inline generates it)
# ---------------------------------------------------------------------
cpp_src = r"""
#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor> fused_forward_cuda(
        torch::Tensor x,
        torch::Tensor weight,
        float   scale);
"""

# ---------------------------------------------------------------------
# Compile and load the extension
# ---------------------------------------------------------------------
fused_kernels = load_inline(
    name="fused_forward_ext_v2",
    cpp_sources=[cpp_src],
    cuda_sources=[source],
    functions=["fused_forward_cuda"],
    verbose=False,
)

# ---------------------------------------------------------------------
# PyTorch module using the custom CUDA kernels
# ---------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Optimised model using custom CUDA kernels.
    Computes:  out[b] = scaling_factor * sum_h(dot(x[b], weight[h]))
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(
            hidden_size, input_size, device='cuda', dtype=torch.float32))
        self.scaling_factor = float(scaling_factor)

    def forward(self, x):
        out, _ = fused_kernels.fused_forward_cuda(
            x.contiguous(), self.weight, self.scaling_factor
        )
        return out
