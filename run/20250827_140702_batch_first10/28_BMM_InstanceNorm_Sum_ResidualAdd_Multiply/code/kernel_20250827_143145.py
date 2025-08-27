import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


CUDA_SRC = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

//
// fused_instance_norm_add_mul
//    1. Instance-normalise the first input along the last dimension
//    2. Add the second input
//    3. Multiply by the second input
//
//    out = (norm(x) + y) * y
//

// CUDA kernel -------------------------------------------------------------
__global__ void fused_kernel(const float* __restrict__ x,
                             const float* __restrict__ y,
                             float* __restrict__ out,
                             const int64_t C,
                             const float eps)
{
    extern __shared__ float sdata[];               // dynamic shared memory
    float* s_sum    = sdata;                       // size blockDim.x
    float* s_sum_sq = sdata + blockDim.x;          // size blockDim.x

    const int n = blockIdx.x;                      // sample index
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    // ---------------------------------------------------------------------
    // 1) compute mean & variance for the nth sample
    float local_sum = 0.f;
    float local_sum_sq = 0.f;

    for (int64_t c = tid; c < C; c += stride) {
        const float val = x[n * C + c];
        local_sum     += val;
        local_sum_sq  += val * val;
    }
    s_sum[tid]    = local_sum;
    s_sum_sq[tid] = local_sum_sq;
    __syncthreads();

    // reduction
    for (int offset = stride >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_sum[tid]    += s_sum[tid + offset];
            s_sum_sq[tid] += s_sum_sq[tid + offset];
        }
        __syncthreads();
    }

    const float mean   = s_sum[0] / static_cast<float>(C);
    const float var    = s_sum_sq[0] / static_cast<float>(C) - mean * mean;
    const float invStd = rsqrtf(var + eps);

    // ---------------------------------------------------------------------
    // 2) normalise + add + mul
    for (int64_t c = tid; c < C; c += stride) {
        const float xn   = (x[n * C + c] - mean) * invStd;  // normalised
        const float valY = y[n * C + c];
        out[n * C + c]   = (xn + valY) * valY;              // fused op
    }
}

// C++ launcher ------------------------------------------------------------
torch::Tensor fused_forward(torch::Tensor x,
                            torch::Tensor y,
                            const float  eps)
{
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(y.is_cuda(), "y must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(y.dtype() == torch::kFloat32, "y must be float32");
    TORCH_CHECK(x.sizes() == y.sizes(),
                "x and y must have the same shape");

    const auto N = x.size(0);
    const auto C = x.size(1);

    auto out = torch::empty_like(x);

    const int block = 256;
    const dim3 grid(N);
    const size_t shmem = block * sizeof(float) * 2;

    fused_kernel<<<grid, block, shmem>>>(x.data_ptr<float>(),
                                         y.data_ptr<float>(),
                                         out.data_ptr<float>(),
                                         C,
                                         eps);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_forward", &fused_forward, "fused instance-norm + add + mul (CUDA)");
}
"""

# compile & load
_fused = load_inline(name="fused_instance_add_mul",
                     cpp_sources="",
                     cuda_sources=CUDA_SRC,
                     functions=["fused_forward"],
                     verbose=False)


class ModelNew(nn.Module):
    """
    Optimised model:
        Linear  ->  fused(instance_norm + add + mul)
    """
    def __init__(self, in_features, out_features, eps=1e-5):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.eps = eps

    def forward(self, x, y):
        x = self.linear(x)
        # tensors must be contiguous & CUDA
        x = x.contiguous().cuda()
        y = y.contiguous().cuda()
        return _fused.fused_forward(x, y, self.eps)


# helpers ------------------------------------------------------------------
batch_size   = 512
in_features  = 4096
out_features = 4096

def get_inputs():
    return [torch.rand(batch_size, in_features, device="cuda"),
            torch.rand(batch_size, out_features, device="cuda")]

def get_init_inputs():
    return [in_features, out_features]
