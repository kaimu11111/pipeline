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
 *   X layout : (B, inF)            row–major
 *   W layout : (outF, inF)         row–major
 *   Y layout : (B, outF)           row–major
 *
 * The previous implementation produced large numerical errors because it
 * re-used an `acc` register across different `k0` tiles without zero-initialising
 * it for each batch-loop iteration when `TILE_K` did not divide `inF`.
 * We fix this by moving the declaration/initialisation of `acc` *inside* the
 * inner `k0` loop, then doing a final reduction into a separate scalar before
 * writing the result.  This guarantees that any out-of-range reads contribute
 * exactly `0.f`, eliminating the accumulation of stale values.
 */
template<int TILE_K>
__global__ void linear_forward_kernel(const float* __restrict__ X,
                                      const float* __restrict__ W,
                                      const float* __restrict__ bias,
                                      float* __restrict__ Y,
                                      const int B,
                                      const int inF,
                                      const int outF)
{
    const int o = blockIdx.x * blockDim.x + threadIdx.x;  // output feature
    if (o >= outF) return;

    // Iterate over the batch dimension with a stride of gridDim.y
    for (int b = blockIdx.y; b < B; b += gridDim.y)
    {
        float y_val = 0.f;                     // <─ final accumulator

        // Pointer to beginning of this (b,k) slice in X
        const float* x_ptr = X + b * inF;
        const float* w_ptr = W + o * inF;

        // Loop over input features in TILE_K chunks
        for (int k0 = 0; k0 < inF; k0 += TILE_K)
        {
            float acc = 0.f;                   // <─ tile-local accumulator
            float x_frag[TILE_K];              //    cached X[b, k0 : k0+TILE_K)

            // Load a tile of X into registers
            #pragma unroll
            for (int t = 0; t < TILE_K; ++t)
            {
                const int k = k0 + t;
                x_frag[t] = (k < inF) ? x_ptr[k] : 0.f;
            }

            // Multiply-accumulate with corresponding W tile
            #pragma unroll
            for (int t = 0; t < TILE_K; ++t)
            {
                const int k = k0 + t;
                if (k < inF)
                    acc += x_frag[t] * w_ptr[k];
            }
            y_val += acc;                      // accumulate tile result
        }

        // Add bias and write out
        y_val += bias[o];
        Y[b * outF + o] = y_val;
    }
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

    const int64_t B    = X.size(0);
    const int64_t inF  = X.size(1);
    const int64_t outF = W.size(0);

    TORCH_CHECK(W.size(1) == inF,        "W shape mismatch");
    TORCH_CHECK(bias.numel() == outF,    "bias shape mismatch");

    auto Y = torch::empty({B, outF}, X.options());

    constexpr int BLOCK_X = 128;                      // threads per block (x-dim)
    const int grid_x = (outF + BLOCK_X - 1) / BLOCK_X;
    const int grid_y = std::min<int64_t>(B, 65535);   // avoid CUDA limit

    const dim3 block(BLOCK_X);
    const dim3 grid(grid_x, grid_y);

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
    name="model_cuda_kernels_fixed",
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
        self.linear = nn.Linear(in_features, out_features, bias=True).cuda()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            x = x.cuda(non_blocking=True)
        x = x.contiguous()
        y = model_cuda.model_fwd_cuda(
            x,
            self.linear.weight.contiguous(),  # (outF, inF)
            self.linear.bias.contiguous(),
        )
        return y
