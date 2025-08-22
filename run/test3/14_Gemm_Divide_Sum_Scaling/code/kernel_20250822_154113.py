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
    const int h = blockIdx.x;   // hidden index
    const int b = blockIdx.y;   // batch  index

    acc_t sum = static_cast<acc_t>(0);

    // loop over the input dimension – ­each thread handles a strided
    // subset of the vector; no shared-memory reduction is necessary
    // because we are using warp-wide reduction with shuffles.
    for (int idx = threadIdx.x; idx < input_size; idx += BLOCK_SIZE) {
        sum += static_cast<acc_t>(x[b * input_size + idx]) *
               static_cast<acc_t>(w[h * input_size + idx]);
    }

    // warp-wide reduction ------------------------------------------------
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // only one thread per warp writes – we launch one warp per block,
    // so threadIdx.x == 0 is sufficient
    if (threadIdx.x == 0) {
        out[b * hidden_size + h] = sum * scale;
    }
}

std::tuple<torch::Tensor, torch::Tensor> fused_forward_cuda(
        torch::Tensor x,
        torch::Tensor weight,
        const double  scale_double) {

    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "x & weight must be CUDA tensors");
    TORCH_CHECK(x.dim() == 2 && weight.dim() == 2, "x / w must be 2-D");
    TORCH_CHECK(x.size(1) == weight.size(1),
                "x.shape[1] must equal weight.shape[1] (input_size)");

    const int batch_size  = x.size(0);
    const int input_size  = x.size(1);
    const int hidden_size = weight.size(0);

    // we always accumulate in float – this is enough for fp16/bf16 numerics
    using acc_t = float;
    constexpr int BLOCK = 256;

    auto out = torch::empty({batch_size, hidden_size},
                            x.options().dtype(torch::kFloat));

    const dim3 grid(hidden_size, batch_size);
    const size_t shm_bytes = 0;               // using warp shuffles only
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "linear_kernel", ([&] {
        linear_kernel<scalar_t, acc_t, BLOCK><<<grid, BLOCK, shm_bytes, stream>>>(
            x.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            out.data_ptr<acc_t>(),
            static_cast<acc_t>(scale_double),
            batch_size, hidden_size, input_size);
    }));

    // The second tensor is kept for API compatibility with the original
    // reference implementation.
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
    name="fused_forward_ext_v4_fixed",
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
        self.weight = nn.Parameter(torch.randn(
            hidden_size, input_size, dtype=dtype, device="cuda"))
        self.scaling_factor = float(scaling_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are contiguous for the custom kernel
        out, _ = fused_kernels.fused_forward_cuda(
            x.contiguous(), self.weight.contiguous(), self.scaling_factor
        )
        # match the input dtype (fp16/bf16 needs cast-down from fp32 accumulator)
        return out.to(dtype=x.dtype)
