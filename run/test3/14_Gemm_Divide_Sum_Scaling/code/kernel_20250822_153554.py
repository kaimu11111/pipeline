import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------
# CUDA source code
# ---------------------------------------------------------------------
source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

template <typename scalar_t, typename acc_t, int BLOCK_SIZE>
__global__ void linear_kernel(const scalar_t* __restrict__ x,
                              const scalar_t* __restrict__ w,
                              acc_t*          __restrict__ out,
                              const acc_t                 scale,
                              const int                   batch_size,
                              const int                   hidden_size,
                              const int                   input_size) {
    const int b = blockIdx.y;   // batch index
    const int h = blockIdx.x;   // hidden index

    extern __shared__ acc_t shm[];
    acc_t accum = 0;

    // stride over input dimension
    for (int idx = threadIdx.x; idx < input_size; idx += BLOCK_SIZE) {
        const acc_t xv = static_cast<acc_t>(x[b * input_size + idx]);
        const acc_t wv = static_cast<acc_t>(w[h * input_size + idx]);
        accum += xv * wv;
    }
    shm[threadIdx.x] = accum;
    __syncthreads();

    // intra-block reduction
    for (int s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) shm[threadIdx.x] += shm[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out[b * hidden_size + h] = shm[0] * scale;
    }
}

std::tuple<torch::Tensor, torch::Tensor> fused_forward_cuda(
        torch::Tensor x,
        torch::Tensor weight,
        const double  scale_double) {

    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "Tensors must be on CUDA");
    TORCH_CHECK(x.scalar_type() == weight.scalar_type(),
                "x and weight must have the same dtype");
    TORCH_CHECK(x.size(1) == weight.size(1),
                "Input dimension mismatch between x and weight");

    const int batch_size  = x.size(0);
    const int input_size  = x.size(1);
    const int hidden_size = weight.size(0);

    // choose accumulation type
    using acc_t = float;

    auto out  = torch::empty({batch_size, hidden_size},
                             x.options().dtype(torch::kFloat));

    const int BLOCK = 256;
    const dim3 grid(hidden_size, batch_size);
    const size_t shm_bytes = BLOCK * sizeof(acc_t);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "linear_kernel_launch", ([&] {
        linear_kernel<scalar_t, acc_t, BLOCK><<<grid, BLOCK, shm_bytes, stream>>>(
            x.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            out.data_ptr<acc_t>(),
            static_cast<acc_t>(scale_double),
            batch_size, hidden_size, input_size);
    }));

    // second tensor kept for API compatibility (unused by Python wrapper)
    auto dummy = torch::empty({0}, x.options());

    return std::make_tuple(out, dummy);
}
"""

# ---------------------------------------------------------------------
# C++ declaration prototypes
# ---------------------------------------------------------------------
cpp_src = r"""
#include <torch/extension.h>
std::tuple<torch::Tensor, torch::Tensor> fused_forward_cuda(
        torch::Tensor x,
        torch::Tensor weight,
        double  scale_double);
"""

# ---------------------------------------------------------------------
# Compile & load
# ---------------------------------------------------------------------
fused_kernels = load_inline(
    name="fused_forward_ext_v4",
    cpp_sources=[cpp_src],
    cuda_sources=[source],
    functions=["fused_forward_cuda"],
    verbose=False,
)

# ---------------------------------------------------------------------
# PyTorch module wrapper
# ---------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor, dtype=torch.float32):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size,
                                               device='cuda', dtype=dtype))
        self.scaling_factor = float(scaling_factor)

    def forward(self, x):
        out, _ = fused_kernels.fused_forward_cuda(
            x.contiguous(), self.weight.contiguous(), self.scaling_factor
        )
        return out.to(x.dtype)
