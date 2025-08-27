import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

/////////////////////////////////////////////////////////////////
// helpers for warp & block reduction
/////////////////////////////////////////////////////////////////
template <typename T>
__inline__ __device__ T warpReduceSum(T val){
    for (int offset = warpSize/2; offset > 0; offset >>= 1){
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename T>
__inline__ __device__ T warpReduceMax(T val){
    for (int offset = warpSize/2; offset > 0; offset >>= 1){
        T other = __shfl_down_sync(0xffffffff, val, offset);
        val     = val > other ? val : other;
    }
    return val;
}

template <typename T>
__inline__ __device__ T blockReduceSum(T val){
    static __shared__ T shared[32];  // 32 warps max within a block (1024 threads)
    int lane = threadIdx.x & 31;     // 0..31
    int wid  = threadIdx.x >> 5;     // warp id
    val      = warpReduceSum(val);   // each warp has its local sum
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    // reduce sums of warps
    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : (T)0;
    if (wid == 0){
        val = warpReduceSum(val);
    }
    return val;
}

template <typename T>
__inline__ __device__ T blockReduceMax(T val){
    static __shared__ T shared[32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    val      = warpReduceMax(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : -FLT_MAX;
    if (wid == 0){
        val = warpReduceMax(val);
    }
    return val;
}

/////////////////////////////////////////////////////////////////
// Kernel 1 : fused scale, double, clamp (element-wise)
/////////////////////////////////////////////////////////////////
__global__ void scale_double_clamp_kernel(
        const float* __restrict__ in,
        float* __restrict__ out,
        const float scale,
        const float clamp_min,
        const float clamp_max,
        const size_t size){
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size){
        float val = in[idx] * scale * 2.0f;
        val       = fminf(fmaxf(val, clamp_min), clamp_max);
        out[idx]  = val;
    }
}

/////////////////////////////////////////////////////////////////
// Kernel 2 : row-wise logsumexp + mish and final product
//            output[i] = LSE_i * mish(LSE_i)
/////////////////////////////////////////////////////////////////
__global__ void row_logsumexp_mish_mul_kernel(
        const float* __restrict__ in,
        float* __restrict__ out,
        const int hidden){
    const int row = blockIdx.x;                // one block per row
    const int tid = threadIdx.x;
    const int row_start = row * hidden;

    // 1) compute max for numerical stability
    float local_max = -FLT_MAX;
    for (int idx = tid; idx < hidden; idx += blockDim.x){
        float v = in[row_start + idx];
        local_max = v > local_max ? v : local_max;
    }
    float max_val = blockReduceMax(local_max);

    // 2) compute sum of exp(x - max)
    float local_sum = 0.f;
    for (int idx = tid; idx < hidden; idx += blockDim.x){
        float v = in[row_start + idx];
        local_sum += __expf(v - max_val);
    }
    float sum_exp = blockReduceSum(local_sum);

    if (tid == 0){
        float lse = logf(sum_exp) + max_val;               // logsumexp
        // mish(x) = x * tanh(softplus(x))
        float softplus = log1pf(expf(lse));
        float mish     = lse * tanhf(softplus);
        out[row]       = lse * mish;                       // final product
    }
}

/////////////////////////////////////////////////////////////////
// C++/PyTorch interfaces
/////////////////////////////////////////////////////////////////
torch::Tensor scale_double_clamp_cuda(torch::Tensor input,
                                      float scale,
                                      float clamp_min,
                                      float clamp_max){
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    auto output = torch::empty_like(input);
    const size_t N = input.numel();
    const int block = 256;
    const int grid  = (N + block - 1) / block;
    scale_double_clamp_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        scale,
        clamp_min,
        clamp_max,
        N);
    return output;
}

torch::Tensor row_logsumexp_mish_mul_cuda(torch::Tensor input){
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "expect 2-D tensor");
    const int batch  = input.size(0);
    const int hidden = input.size(1);
    auto output = torch::empty({batch, 1}, input.options());
    const int block = 256;
    row_logsumexp_mish_mul_kernel<<<batch, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        hidden);
    return output;
}

/////////////////////////////////////////////////////////////////
// bindings
/////////////////////////////////////////////////////////////////
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scale_double_clamp_cuda", &scale_double_clamp_cuda, "scale*2 + clamp (CUDA)");
    m.def("row_logsumexp_mish_mul_cuda", &row_logsumexp_mish_mul_cuda, "row logsumexp * mish (CUDA)");
}
"""

cpp_decls = """
torch::Tensor scale_double_clamp_cuda(torch::Tensor input,
                                      float scale,
                                      float clamp_min,
                                      float clamp_max);

torch::Tensor row_logsumexp_mish_mul_cuda(torch::Tensor input);
"""

# compile & load
fusion_ops = load_inline(name='fusion_ops',
                         cpp_sources=cpp_decls,
                         cuda_sources=cuda_source,
                         functions=['scale_double_clamp_cuda',
                                    'row_logsumexp_mish_mul_cuda'],
                         verbose=False)

############################################
# Optimised Python module
############################################
class ModelNew(nn.Module):
    """
    Optimised model that re-implements the original computation with
    two custom CUDA kernels:
        1. fused scale *2 and clamp (element-wise)
        2. row-wise logsumexp followed by Mish activation and product
    The linear layer remains the highly-optimised cuBLAS matmul.
    """
    def __init__(self, input_size, hidden_size, scale_factor,
                 clamp_min, clamp_max):
        super().__init__()
        self.matmul = nn.Linear(input_size, hidden_size, bias=True).cuda()
        self.scale_factor = float(scale_factor)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

    def forward(self, x):
        # 1) linear projection
        x = self.matmul(x)
        # 2) fused scale *2 and clamp
        x = fusion_ops.scale_double_clamp_cuda(
                x,
                self.scale_factor,
                self.clamp_min,
                self.clamp_max)
        # 3) row-wise logsumexp then mish product
        x = fusion_ops.row_logsumexp_mish_mul_cuda(x)
        return x
