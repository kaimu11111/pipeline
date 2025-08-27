import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------------
# CUDA kernel: computes, for every row of X,      out[row] = X[row]·w_sum + b_sum
# ---------------------------------------------------------------------------
cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void row_dot_kernel(const float* __restrict__ x,
                               const float* __restrict__ w_sum,
                               float* __restrict__ out,
                               const float  b_sum,
                               const int    in_feats)
{
    extern __shared__ float sdata[];
    const int  row    = blockIdx.x;          // one block per input row
    const int  tid    = threadIdx.x;
    const int  stride = blockDim.x;

    const float* x_row = x + row * in_feats;

    // --- parallel dot ------------------------------------------------------
    float thread_sum = 0.f;
    for (int i = tid; i < in_feats; i += stride)
        thread_sum += x_row[i] * w_sum[i];

    sdata[tid] = thread_sum;
    __syncthreads();

    // --- intra–block reduction --------------------------------------------
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        out[row] = sdata[0] + b_sum;
}

torch::Tensor row_dot_cuda(torch::Tensor x,
                           torch::Tensor w_sum,
                           const float   b_sum)
{
    TORCH_CHECK(x.is_cuda(),     "x must reside on CUDA");
    TORCH_CHECK(w_sum.is_cuda(), "w_sum must reside on CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32 &&
                w_sum.dtype() == torch::kFloat32,
                "supports float32 tensors only");

    const int batch      = x.size(0);
    const int in_feats   = x.size(1);
    auto out = torch::empty({batch}, x.options());

    constexpr int BLOCK = 256;
    const dim3 grid(batch);
    const size_t shmem = BLOCK * sizeof(float);

    row_dot_kernel<<<grid, BLOCK, shmem>>>(x.data_ptr<float>(),
                                           w_sum.data_ptr<float>(),
                                           out.data_ptr<float>(),
                                           b_sum,
                                           in_feats);
    return out;
}
"""

cpp_hdr = "torch::Tensor row_dot_cuda(torch::Tensor, torch::Tensor, float);"

# compile & load
row_dot = load_inline(name       = "row_dot",
                      cpp_sources  = cpp_hdr,
                      cuda_sources = cuda_src,
                      functions    = ["row_dot_cuda"],
                      verbose      = False)

# ---------------------------------------------------------------------------
# Optimised model
# ---------------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Fused implementation of the original workflow:
        linear -> sum -> max -> mean -> logsumexp -> logsumexp
    Algebraic simplification yields:
        output = sum_j (x W^T + b)_j  (per batch sample)
               = x · (sum_j W_j) + sum_j b_j
    The custom CUDA kernel computes this in one pass.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.empty(out_features))

        # default initialisation identical to nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # compute column-wise weight sum and bias sum (cheap, small tensors)
        w_sum = self.weight.sum(dim=0).contiguous()
        b_sum = self.bias.sum().item()

        # invoke custom CUDA kernel
        out = row_dot.row_dot_cuda(x.contiguous(), w_sum, b_sum)

        # keep original shape: (batch, 1)
        return out.unsqueeze(1)


# ---------------------------------------------------------------------------
# Helper functions identical to the original file
# ---------------------------------------------------------------------------
batch_size   = 512
in_features  = 4096
out_features = 4096

def get_inputs():
    return [torch.rand(batch_size, in_features, device='cuda', dtype=torch.float32)]

def get_init_inputs():
    return [in_features, out_features]
