import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Inline CUDA/C++ source for fused BatchNorm + scale + Softmax
fused_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// First kernel: apply BN + scale + exp, and compute partial row sums (for softmax)
__global__ void bn_scale_exp_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float  scale,
    const int    rows,
    const int    cols,
    const float  eps,
    float* __restrict__ row_sum
) {
    // One block per row, each block has blockDim.x threads
    int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float sdata[]; 
    float partial_sum = 0.0f;

    // Loop over columns in strides of blockDim.x
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        // BN transform:
        // val = scale * [ gamma[col] * ((x - mean[col]) / sqrt(var[col] + eps)) + beta[col] ]
        float normed = (x[row * cols + col] - running_mean[col]) 
                       / sqrtf(running_var[col] + eps);
        float val = scale * (gamma[col] * normed + beta[col]);
        
        // exponent for softmax
        val = expf(val);

        // store the result
        out[row * cols + col] = val;
        partial_sum += val;
    }

    // reduce partial_sum within the block
    sdata[threadIdx.x] = partial_sum;
    __syncthreads();

    // blockDim.x-way reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // thread 0 writes the block-level sum to row_sum
    if (threadIdx.x == 0) {
        row_sum[row] = sdata[0];
    }
}

// Second kernel: divide by the row sum (to finish softmax)
__global__ void softmax_div_kernel(
    float* __restrict__ out,
    const float* __restrict__ row_sum,
    const int rows,
    const int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        out[row * cols + col] /= row_sum[row];
    }
}

// C++ interface
torch::Tensor fused_bn_scale_softmax_cuda(
    torch::Tensor x,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor scale_t,
    float eps
) {
    // x: [rows, cols]
    // running_mean, running_var, weight, bias: [cols]
    // scale_t: [1] or scalar
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(running_mean.is_cuda(), "running_mean must be a CUDA tensor");
    TORCH_CHECK(running_var.is_cuda(), "running_var must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(scale_t.is_cuda(), "scale must be a CUDA tensor");

    int rows = x.size(0);
    int cols = x.size(1);

    auto out = torch::empty_like(x);
    auto row_sum = torch::empty({rows}, x.options());

    float scale_val = scale_t.data_ptr<float>()[0];

    // Launch bn_scale_exp_kernel
    dim3 grid(rows);
    dim3 block(256);
    size_t shared_mem_size = 256 * sizeof(float);

    bn_scale_exp_kernel<<<grid, block, shared_mem_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        scale_val,
        rows,
        cols,
        eps,
        row_sum.data_ptr<float>()
    );

    // Launch softmax_div_kernel
    softmax_div_kernel<<<grid, block>>>(
        out.data_ptr<float>(),
        row_sum.data_ptr<float>(),
        rows,
        cols
    );

    return out;
}
"""

# C++ declaration of the above function
fused_cpp_src = r"""
torch::Tensor fused_bn_scale_softmax_cuda(
    torch::Tensor x,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor scale_t,
    float eps
);
"""

# Compile and load the inline extension
fused_ops = load_inline(
    name="fused_bn_scale_softmax",
    cpp_sources=fused_cpp_src,
    cuda_sources=fused_source,
    functions=["fused_bn_scale_softmax_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["--use_fast_math", "-O3"],
)


class ModelNew(nn.Module):
    """
    Optimized Model that uses a custom CUDA fused kernel for BatchNorm + scale + Softmax.
    The GEMM (linear layer) remains the default PyTorch implementation.
    """
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.bn_eps = bn_eps

    def forward(self, x):
        # Perform linear layer (matmul + bias)
        x = self.gemm(x)
        # Fused BN + scale + Softmax
        x = fused_ops.fused_bn_scale_softmax_cuda(
            x,
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.weight,
            self.bn.bias,
            self.scale,
            self.bn_eps
        )
        return x
