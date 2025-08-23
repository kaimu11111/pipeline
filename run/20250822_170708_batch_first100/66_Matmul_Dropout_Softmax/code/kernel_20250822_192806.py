import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>

// ---------------------------------------------
// Naive Matrix Multiplication
// A: (M x K), B: (K x N), C: (M x N)
// bias: (N)
// ---------------------------------------------
__global__ void matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ C,
    int M, int K, int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float val = 0.0f;
        for (int i = 0; i < K; i++) {
            val += A[row * K + i] * B[i * N + col];
        }
        val += bias[col];
        C[row * N + col] = val;
    }
}

// ---------------------------------------------
// Naive Dropout (in-place-like, but we create an out tensor)
//   out[i] = in[i] * mask[i] / (1 - p)
//   mask[i] = (rand() >= p) ? 1.0 : 0.0
// ---------------------------------------------
__global__ void dropout_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    float p,
    unsigned long long seed,
    int total_elems
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elems) {
        // Each thread gets its own RNG state
        curandStatePhilox4_32_10_t state;
        curand_init(seed, idx, 0, &state);

        float rand_val = curand_uniform(&state);
        float mask = (rand_val >= p) ? 1.0f : 0.0f;
        float scale = 1.0f / (1.0f - p);

        out[idx] = in[idx] * mask * scale;
    }
}

// ---------------------------------------------
// Naive Softmax over dim=1: 
//   input shape: (M x N)
//   For each row, out = exp(in - max) / sum(exp(...))
// ---------------------------------------------
__global__ void softmax_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    int M, 
    int N
) {
    // One row per block
    int row = blockIdx.x;
    if (row < M) {
        // Compute max
        float max_val = -FLT_MAX;
        for(int col = 0; col < N; col++){
            float val = in[row * N + col];
            if(val > max_val){
                max_val = val;
            }
        }
        // Compute sum of exp
        float sum_val = 0.0f;
        for(int col = 0; col < N; col++){
            sum_val += expf(in[row * N + col] - max_val);
        }
        // Write softmax
        for(int col = 0; col < N; col++){
            out[row * N + col] = expf(in[row * N + col] - max_val) / sum_val;
        }
    }
}

// Wrapper: matmul
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor bias) {
    // A: (M x K), B: (K x N), bias: (N)
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(1);

    auto opts = torch::TensorOptions().dtype(A.dtype()).device(A.device());
    auto C = torch::empty({M, N}, opts);

    dim3 block(16,16);
    dim3 grid((N + block.x - 1)/block.x, (M + block.y - 1)/block.y);

    matmul_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        bias.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, N
    );
    return C;
}

// Wrapper: dropout
torch::Tensor dropout_cuda(torch::Tensor in, float p, unsigned long long seed) {
    auto out = torch::empty_like(in);
    int total_elems = in.numel();

    int blockSize = 256;
    int gridSize = (total_elems + blockSize - 1) / blockSize;

    dropout_kernel<<<gridSize, blockSize>>>(
        in.data_ptr<float>(),
        out.data_ptr<float>(),
        p,
        seed,
        total_elems
    );
    return out;
}

// Wrapper: softmax
torch::Tensor softmax_cuda(torch::Tensor in) {
    int64_t M = in.size(0);
    int64_t N = in.size(1);

    auto out = torch::empty_like(in);
    // Launch M blocks, 1D block
    softmax_kernel<<<M, 1>>>(
        in.data_ptr<float>(),
        out.data_ptr<float>(),
        M, N
    );
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_cuda", &matmul_cuda, "Naive Matmul (CUDA)");
    m.def("dropout_cuda", &dropout_cuda, "Naive Dropout (CUDA)");
    m.def("softmax_cuda", &softmax_cuda, "Naive Softmax (CUDA)");
}
"""

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources="torch::Tensor matmul_cuda(torch::Tensor, torch::Tensor, torch::Tensor);\n"
                "torch::Tensor dropout_cuda(torch::Tensor, float, unsigned long long);\n"
                "torch::Tensor softmax_cuda(torch::Tensor);\n",
    cuda_sources=source,
    extra_include_paths=[],
    extra_cflags=[],
    extra_ldflags=["-lcuda","-lcudart","-lcurand"],
    functions=["matmul_cuda", "dropout_cuda", "softmax_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized Model with custom CUDA kernels for matmul, dropout, and softmax.
    """
    def __init__(self, in_features, out_features, dropout_p):
        super(ModelNew, self).__init__()
        # Match original linear's logic: weight shape (out_features, in_features)
        # but we want shape (in_features, out_features) for a direct (x * W).
        # We'll store as (in_features, out_features) and handle bias with size (out_features).
        self.weight = nn.Parameter(torch.randn(in_features, out_features, device='cuda') * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features, device='cuda'))
        self.dropout_p = dropout_p

    def forward(self, x):
        # shape of x: (batch_size, in_features)
        # weight: (in_features, out_features)
        # => matmul result shape: (batch_size, out_features)
        out = fused_ops.matmul_cuda(x, self.weight, self.bias)
        # dropout
        # seed can be any value, here we just pick a constant or use torch.randint
        seed = 42
        out = fused_ops.dropout_cuda(out, self.dropout_p, seed)
        # softmax
        out = fused_ops.softmax_cuda(out)
        return out
