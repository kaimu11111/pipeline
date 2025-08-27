import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ----------------------------------------------------------------------
# Custom CUDA kernels (inline compilation)
# ----------------------------------------------------------------------
cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

////////////////////////////////////////////////////////////////////////////////
// Kernel 1: column–wise reduction of `weight`
// Computes wsum[j] = sum_i weight[i, j]
////////////////////////////////////////////////////////////////////////////////
__global__ void weight_col_sum_kernel(const float* __restrict__ weight,
                                      float* __restrict__ wsum,
                                      int hidden_size,
                                      int input_size) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= input_size) return;

    float sum = 0.0f;
    for (int row = 0; row < hidden_size; ++row) {
        sum += weight[row * input_size + col];
    }
    wsum[col] = sum;
}

////////////////////////////////////////////////////////////////////////////////
// Kernel 2: batched dot product
// Computes out[b] = scale * sum_j x[b,j] * wsum[j]
////////////////////////////////////////////////////////////////////////////////
__global__ void batched_dot_kernel(const float* __restrict__ x,
                                   const float* __restrict__ wsum,
                                   float* __restrict__ out,
                                   int batch_size,
                                   int input_size,
                                   float scale) {
    int row = blockIdx.x;                 // One block per batch sample
    int tid = threadIdx.x;
    float partial = 0.0f;

    // Strided loop across the vector length
    for (int col = tid; col < input_size; col += blockDim.x) {
        partial += x[row * input_size + col] * wsum[col];
    }

    // Shared-memory reduction within the block
    __shared__ float shared[256];
    shared[tid] = partial;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[row] = shared[0] * scale;
    }
}

////////////////////////////////////////////////////////////////////////////////
// C++/CUDA front-end (called from Python)
////////////////////////////////////////////////////////////////////////////////
torch::Tensor fused_forward_cuda(torch::Tensor x,
                                 torch::Tensor weight,
                                 double scaling_factor) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");

    const int batch_size  = x.size(0);
    const int input_size  = x.size(1);
    const int hidden_size = weight.size(0);

    // ------------------------------------------------------------------
    // Step 1: column-wise reduction of the weight matrix
    // ------------------------------------------------------------------
    auto wsum = torch::empty({input_size}, x.options());
    const int block_size = 256;
    int grid_sum = (input_size + block_size - 1) / block_size;
    weight_col_sum_kernel<<<grid_sum, block_size>>>(
        weight.data_ptr<float>(),
        wsum.data_ptr<float>(),
        hidden_size,
        input_size);

    // ------------------------------------------------------------------
    // Step 2: batched dot product with scaling (scale = 0.5 * scaling_factor)
    // ------------------------------------------------------------------
    auto out = torch::zeros({batch_size}, x.options());
    float scale = static_cast<float>(scaling_factor * 0.5);
    int grid_dot = batch_size;  // one block per sample
    batched_dot_kernel<<<grid_dot, block_size>>>(
        x.data_ptr<float>(),
        wsum.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        input_size,
        scale);

    // Return shape: (batch_size, 1)
    return out.view({batch_size, 1});
}
"""

cpp_decl = r"""
torch::Tensor fused_forward_cuda(torch::Tensor x,
                                 torch::Tensor weight,
                                 double scaling_factor);
"""

fused_ops = load_inline(
    name="fused_matmul_div_sum_scale",
    cpp_sources=cpp_decl,
    cuda_sources=cuda_src,
    functions=["fused_forward_cuda"],
    verbose=False,
)

# ----------------------------------------------------------------------
# Optimised model
# ----------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Optimised version of the original model using fused custom CUDA kernels.
    The fused kernel computes:
        y = scaling_factor * 0.5 * sum_k (x @ weight.T)[k]
    which is mathematically equivalent to the original sequence:
        matmul -> divide by 2 -> sum -> scale
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size, device="cuda"))
        self.scaling_factor = float(scaling_factor)  # ensure Python float
        self.fused_forward = fused_ops.fused_forward_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure contiguous memory layout for kernel inputs
        return self.fused_forward(
            x.contiguous(), self.weight.contiguous(), self.scaling_factor
        )
