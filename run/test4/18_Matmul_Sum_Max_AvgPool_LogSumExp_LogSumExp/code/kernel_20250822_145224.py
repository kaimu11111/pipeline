import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------------
# CUDA source: kernels + host wrapper
# ------------------------------------------------------------------
source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/* -------- CUDA error checking -------- */
#define CUDA_CHECK_ERRORS()                                                   \
  do {                                                                        \
    cudaError_t err = cudaGetLastError();                                     \
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed : ",                  \
                cudaGetErrorString(err));                                     \
  } while (0)

/*
 * Kernel
 *   Computes:  Y[b, o] = Σ_k  X[b, k] * W[o, k] + bias[o]
 *   X layout: (B, inF)            row–major
 *   W layout: (outF, inF)         row–major
 *   Y layout: (B, outF)           row–major
 */
template<int TILE_K>
__global__ void linear_forward_kernel(const float* __restrict__ X,
                                      const float* __restrict__ W,
                                      const float* __restrict__ bias,
                                      float* __restrict__ Y,
                                      int B, int inF, int outF) {
    const int o = blockIdx.x * blockDim.x + threadIdx.x;  // output feature
    const int b = blockIdx.y * blockDim.y + threadIdx.y;  // batch

    if (o >= outF || b >= B) return;

    const float* w_ptr = W + o * inF;
    const float* x_ptr = X + b * inF;

    float acc = 0.f;

    #pragma unroll
    for (int k = 0; k < inF; k += TILE_K) {
        int k_lim = min(TILE_K, inF - k);
        #pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            if (kk < k_lim) {
                acc += x_ptr[k + kk] * w_ptr[k + kk];
            }
        }
    }

    acc += bias[o];
    Y[b * outF + o] = acc;
}

/*
 * Host wrapper exposed to Python
 */
torch::Tensor model_fwd_cuda(torch::Tensor X,
                             torch::Tensor W,
                             torch::Tensor bias) {
    CHECK_INPUT(X);
    CHECK_INPUT(W);
    CHECK_INPUT(bias);

    TORCH_CHECK(X.scalar_type() == torch::kFloat32, "X must be float32");
    TORCH_CHECK(W.scalar_type() == torch::kFloat32, "W must be float32");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "bias must be float32");

    const int64_t B     = X.size(0);
    const int64_t inF   = X.size(1);
    const int64_t outF  = W.size(0);

    TORCH_CHECK(W.size(1) == inF, "W shape mismatch");
    TORCH_CHECK(bias.numel() == outF, "bias shape mismatch");

    auto Y = torch::empty({B, outF}, X.options());

    constexpr int BLOCK_X = 16;
    constexpr int BLOCK_Y = 16;
    const dim3 block(BLOCK_X, BLOCK_Y);
    const dim3 grid((outF + BLOCK_X - 1) / BLOCK_X,
                    (B     + BLOCK_Y - 1) / BLOCK_Y);

    constexpr int TILE_K = 8;
    linear_forward_kernel<TILE_K><<<grid, block>>>(
        X.data_ptr<float>(),
        W.data_ptr<float>(),
        bias.data_ptr<float>(),
        Y.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(inF),
        static_cast<int>(outF));

    CUDA_CHECK_ERRORS();
    return Y;
}
"""

# ------------------------------------------------------------------
# C++ declaration(s) to expose
# ------------------------------------------------------------------
cpp_src = """
torch::Tensor model_fwd_cuda(torch::Tensor X,
                             torch::Tensor W,
                             torch::Tensor bias);
"""

# ------------------------------------------------------------------
# Compile & load kernels
# ------------------------------------------------------------------
model_cuda = load_inline(
    name="model_cuda_kernels_v3",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["model_fwd_cuda"],
    verbose=False,
)

# ------------------------------------------------------------------
# Optimised PyTorch module using the custom CUDA kernels
# ------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features).cuda()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            x = x.cuda(non_blocking=True)
        x = x.contiguous()

        y = model_cuda.model_fwd_cuda(
            x,
            self.linear.weight.contiguous(),
            self.linear.bias.contiguous(),
        )
        return y
