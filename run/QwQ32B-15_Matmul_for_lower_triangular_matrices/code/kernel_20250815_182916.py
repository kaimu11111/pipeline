# <your corrected code>
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TS 32  // TileSize

__global__ void matmul_tril_kernel(
    float *C, const float* A, const float* B, const int N) {
    
    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;

    if (bx > by) return;  // Skip blocks above diagonal

    __shared__ float shared_A[TS][TS];
    __shared__ float shared_B[TS][TS];

    unsigned int row = by * TS + ty;
    unsigned int col = bx * TS + tx;
    bool compute_needed = (row < N && col < N && row >= col);
    float sum = 0.0f;

    const int num_phases = (N + TS - 1) / TS;

    for (int phase = 0; phase < num_phases; ++phase) {
        // Load A tile (no transpose)
        unsigned int a_row = by * TS + ty;
        unsigned int a_col = phase * TS + tx;
        shared_A[ty][tx] = (a_row < N && a_col < N) ? 
            A[a_row * N + a_col] : 0.0f;

        // Load B tile (transpose for better multiplication)
        unsigned int b_row = phase * TS + ty;
        unsigned int b_col = bx * TS + tx;
        shared_B[tx][ty] = (b_row < N && b_col < N) ? 
            B[b_row * N + b_col] : 0.0f;

        __syncthreads();

        // Multiply-accumulate
        if (compute_needed) {
            #pragma unroll
            for (int k = 0; k < TS; ++k) {
                sum += shared_A[ty][k] * shared_B[k][tx];
            }
        }
        // Removed incorrect __syncthreads() here
    }

    // Apply tril mask and write
    if (row < N && col < N) {
        C[row * N + col] = compute_needed ? sum : 0.0f;
    }
}

torch::Tensor matmul_tril_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    TORCH_CHECK(A.sizes() == B.sizes(), "Inputs must have matching sizes");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Tensors must be 2D");
    TORCH_CHECK(A.device() == B.device(), "Tensors must be on same device");

    auto C = torch::empty({N, N}, torch::dtype(A.dtype()).device(A.device()));

    const int TileSize = TS;
    int blocksPerDim = (N + TileSize - 1) / TileSize;
    dim3 block(TileSize, TileSize);
    dim3 grid(blocksPerDim, blocksPerDim);

    matmul_tril_kernel<<<grid, block>>>(C.data_ptr<float>(), A.data_ptr<float>(), B.data_ptr<float>(), N);
    cudaDeviceSynchronize();

    return C;
}
"""

cpp_src = (
    "torch::Tensor matmul_tril_cuda(torch::Tensor A, torch::Tensor B);"
)

matmul_tril = load_inline(
    name="matmul_tril",
    cuda_sources=source,
    cpp_sources=cpp_src,
    functions=["matmul_tril_cuda"],
    verbose=True,
    extra_cuda_cflags=["-use_fast_math -Xptxas -dlcm=cg"]
)

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.matmul_tril = matmul_tril

    def forward(self, A, B):
        return self.matmul_tril.matmul_tril_cuda(A, B)
