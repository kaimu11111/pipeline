import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------
# CUDA source code (type–agnostic, numerically correct)
# ---------------------------------------------------------------------
source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

template <typename scalar_t, typename acc_t>
__global__ void col_sum_kernel(const scalar_t* __restrict__ weight,
                               acc_t* __restrict__ wsum,
                               int hidden_size, int input_size) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= input_size) return;

    acc_t s = 0;
    for (int row = 0; row < hidden_size; ++row) {
        s += static_cast<acc_t>(weight[row * input_size + col]);
    }
    wsum[col] = s;
}

template <typename scalar_t, typename acc_t, unsigned int BLOCK_SIZE>
__global__ void batch_dot_kernel(const scalar_t* __restrict__ x,
                                 const acc_t*    __restrict__ wsum,
                                 acc_t*          __restrict__ out,
                                 acc_t           scale,
                                 int batch_size, int input_size) {
    extern __shared__ acc_t shm[];
    int b = blockIdx.x;                       // one block per batch element

    acc_t local = 0;
    for (int idx = threadIdx.x; idx < input_size; idx += BLOCK_SIZE) {
        local += static_cast<acc_t>(x[b * input_size + idx]) * wsum[idx];
    }
    shm[threadIdx.x] = local;
    __syncthreads();

    for (unsigned int s = BLOCK_SIZE >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) shm[threadIdx.x] += shm[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[b] = shm[0] * scale;
}

std::tuple<torch::Tensor, torch::Tensor> fused_forward_cuda(
        torch::Tensor x,
        torch::Tensor weight,
        double   scale_double) {

    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "Tensors must be on CUDA");
    TORCH_CHECK(x.scalar_type() == weight.scalar_type(),
                "x and weight must have the same dtype");

    const int batch_size  = x.size(0);
    const int input_size  = x.size(1);
    const int hidden_size = weight.size(0);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Choose accumulation type (float for fp16/bf16, same for fp32)
    at::ScalarType st = x.scalar_type();
    at::ScalarType acc_st = (st == at::kFloat) ? at::kFloat : at::kFloat;

    auto wsum = torch::empty({input_size},
                             x.options().dtype(acc_st));
    auto out  = torch::empty({batch_size},
                             x.options().dtype(acc_st));

    const int COL_BLOCK = 256;
    const dim3 grid1((input_size + COL_BLOCK - 1) / COL_BLOCK);
    const dim3 block1(COL_BLOCK);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "col_sum", ([&] {
        using acc_t = float;
        col_sum_kernel<scalar_t, acc_t><<<grid1, block1, 0, stream>>>(
            x.scalar_type() == weight.scalar_type() ?
            weight.data_ptr<scalar_t>() : nullptr,   // silent nvcc warning
            wsum.data_ptr<acc_t>(),
            hidden_size, input_size);
    }));

    const int DOT_BLOCK = 256;
    const dim3 grid2(batch_size);
    const dim3 block2(DOT_BLOCK);
    const size_t shm_bytes = DOT_BLOCK * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "batch_dot", ([&] {
        using acc_t = float;
        batch_dot_kernel<scalar_t, acc_t, DOT_BLOCK><<<grid2, block2, shm_bytes, stream>>>(
            x.data_ptr<scalar_t>(),
            wsum.data_ptr<acc_t>(),
            out.data_ptr<acc_t>(),
            static_cast<acc_t>(scale_double),
            batch_size, input_size);
    }));

    return std::make_tuple(out, wsum);
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
        double   scale_double);
"""

# ---------------------------------------------------------------------
# Compile & load
# ---------------------------------------------------------------------
fused_kernels = load_inline(
    name="fused_forward_ext_v3",
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
            x.contiguous(), self.weight, self.scaling_factor
        )
        return out.to(x.dtype)
