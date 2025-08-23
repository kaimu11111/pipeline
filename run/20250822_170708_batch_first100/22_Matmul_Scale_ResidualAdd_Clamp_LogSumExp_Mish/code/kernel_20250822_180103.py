import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Inline CUDA/C++ sources
source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

__device__ inline float mish(float x) {
    // mish(x) = x * tanh( softplus(x) )
    float sp = log1pf(expf(x));
    float tsp = tanhf(sp);
    return x * tsp;
}

// Fused kernel: matmul (naive) + scale*2 + clamp
// A: (M x K), W: (N x K), B: (N), out: (M x N)
__global__ void fused_matmul_scale_res_clamp_kernel(
    const float* __restrict__ A,
    const float* __restrict__ W,
    const float* __restrict__ B,
    float* __restrict__ out,
    int M, int K, int N,
    float scaleFactor,
    float clampMin,
    float clampMax
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float val = 0.f;
        for (int k = 0; k < K; k++) {
            val += A[row * K + k] * W[col * K + k];
        }
        if (B != nullptr) {
            val += B[col];
        }
        // x = x * scale_factor; then x = x + x -> overall factor = 2 * scale_factor
        val *= (2.f * scaleFactor);
        // clamp
        if (val < clampMin) val = clampMin;
        if (val > clampMax) val = clampMax;
        out[row * N + col] = val;
    }
}

// C++ interface for fused matmul+scale+res+clamp
torch::Tensor fused_matmul_scale_res_clamp_cuda(
    torch::Tensor A,
    torch::Tensor W,
    torch::Tensor B,
    float scaleFactor,
    float clampMin,
    float clampMax
) {
    int M = A.size(0);
    int K = A.size(1);
    int N = W.size(0);

    auto out = torch::empty({M, N}, A.options());

    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    fused_matmul_scale_res_clamp_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        W.data_ptr<float>(),
        (B.numel() ? B.data_ptr<float>() : nullptr),
        out.data_ptr<float>(),
        M, K, N,
        scaleFactor, clampMin, clampMax
    );
    return out;
}

// Kernel for logsumexp along dim=1, then multiplied by mish
// in: (M x N), out: (M x 1)
__global__ void logsumexp_dim1_mish_mul_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    int M, int N
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        float max_val = -FLT_MAX;
        for (int col = 0; col < N; col++) {
            float v = in[row * N + col];
            if (v > max_val) {
                max_val = v;
            }
        }
        float sum_exp = 0.f;
        for (int col = 0; col < N; col++) {
            float v = in[row * N + col];
            sum_exp += expf(v - max_val);
        }
        float lse = logf(sum_exp) + max_val;
        // final: x = x * mish(x)
        out[row] = lse * mish(lse);
    }
}

// C++ interface for logsumexp(dim=1) * mish
torch::Tensor logsumexp_dim1_mish_mul_cuda(torch::Tensor in) {
    int M = in.size(0);
    int N = in.size(1);

    auto out = torch::empty({M, 1}, in.options());
    int block = 256;
    int grid = (M + block - 1) / block;

    logsumexp_dim1_mish_mul_kernel<<<grid, block>>>(
        in.data_ptr<float>(),
        out.data_ptr<float>(),
        M, N
    );
    return out;
}
''';

cpp_src = r'''
torch::Tensor fused_matmul_scale_res_clamp_cuda(
    torch::Tensor A,
    torch::Tensor W,
    torch::Tensor B,
    float scaleFactor,
    float clampMin,
    float clampMax
);

torch::Tensor logsumexp_dim1_mish_mul_cuda(torch::Tensor in);
''';

# Build the custom operators
custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=[cpp_src],
    cuda_sources=[source],
    functions=["fused_matmul_scale_res_clamp_cuda", "logsumexp_dim1_mish_mul_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    """
    Optimized version of the model using custom CUDA kernels for:
    (1) Fused matmul + scale + residual doubling + clamp,
    (2) logsumexp along dim=1, then multiplied by mish.
    """
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.empty(hidden_size, input_size))
        self.bias = nn.Parameter(torch.empty(hidden_size))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.zeros_(self.bias)

        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        x = custom_ops.fused_matmul_scale_res_clamp_cuda(
            x, self.weight, self.bias,
            self.scale_factor, self.clamp_min, self.clamp_max
        )
        x = custom_ops.logsumexp_dim1_mish_mul_cuda(x)
        return x
