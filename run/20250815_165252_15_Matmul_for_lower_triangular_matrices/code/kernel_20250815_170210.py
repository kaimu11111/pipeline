import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matmul_tril_kernel(
    float *C, const float* A, const float* B, const int N) {

    const int TileSize = 16;

    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;

    __shared__ float shared_A[TileSize][TileSize];
    __shared__ float shared_B[TileSize][TileSize];

    unsigned int row = by * TileSize + ty;
    unsigned int col = bx * TileSize + tx;
    float sum = 0.0f;

    for (int phase = 0; phase < (N / TileSize); phase++) {

        // Load A's tile data into shared memory (A rows are fixed, columns are phase's)
        unsigned int a_col = phase * TileSize + tx;
        unsigned int a_row = by * TileSize + ty;
        shared_A[ty][tx] = (a_row < N && a_col < N) ? A[a_row * N + a_col] : 0.0f;

        // Load B's tile data into shared memory (B columns are fixed, rows are phase's)
        unsigned int b_row = phase * TileSize + tx;
        unsigned int b_col = bx * TileSize + ty;
        shared_B[ty][tx] = (b_row < N && b_col < N) ? B[b_row * N + b_col] : 0.0f;

        __syncthreads();

        // Multiply and accumulate
        for (int k = 0; k < TileSize; ++k) {
            sum += shared_A[ty][k] * shared_B[k][tx];
        }

        __syncthreads();
    }

    // Write result to output, applying tril
    if (row < N && col < N) {
        if (row < col) {
            C[row * N + col] = 0.0f;
        } else {
            C[row * N + col] = sum;
        }
    }
}

torch::Tensor matmul_tril_cuda(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    TORCH_CHECK(A.sizes() == B.sizes(), "A and B must be the same size");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Tensors must be 2D");
    TORCH_CHECK(A.device() == B.device(), "Tensors must be on the same device");

    auto C = torch::empty({N, N}, torch::dtype(A.dtype()).device(A.device()));

    const int TileSize = 16;
    int blocksPerDim = (N + TileSize - 1) / TileSize;
    dim3 block(TileSize, TileSize);
    dim3 grid(blocksPerDim, blocksPerDim);

    matmul_tril_kernel<<<grid, block>>>(C.data_ptr<float>(), A.data_ptr<float>(), B.data_ptr<float>(), N);

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
    extra_cflags=[""],
    extra_cuda_cflags=[""]
)

class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.matmul_tril = matmul_tril

    def forward(self, A, B):
        return self.matmul_tril.matmul_tril_cuda(A, B)
