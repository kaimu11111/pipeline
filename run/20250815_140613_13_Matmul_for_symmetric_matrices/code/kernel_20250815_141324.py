import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TS 16

__global__ void matmul_symmetric_kernel(float *C, const float *A, const float *B, int N) {
    __shared__ float sA[TS][TS];
    __shared__ float sB[TS][TS];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float cval = 0.0f;

    for (int k = 0; k < (N / TS); k++) {
        // Load A tile: rows [by*TS + ty], cols [k*TS + tx]
        int a_row = by * TS + ty;
        int a_col = k * TS + tx;
        int a_idx = a_row * N + a_col;
        if (a_row < N && a_col < N) {
            sA[ty][tx] = A[a_idx];
        } else {
            sA[ty][tx] = 0.0f;
        }

        // Load B tile: rows [k*TS + ty], cols [bx*TS + tx]
        int b_row = k * TS + ty;
        int b_col = bx * TS + tx;
        int b_idx = b_row * N + b_col;
        if (b_row < N && b_col < N) {
            sB[ty][tx] = B[b_idx];
        } else {
            sB[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial products
        for (int i = 0; i < TS; i++) {
            cval += sA[ty][i] * sB[i][tx];
        }

        __syncthreads();  // Synchronize after shared mem operation but before next iteration
    }

    // Write result to global memory
    int row = by * TS + ty;
    int col = bx * TS + tx;
    if (row < N && col < N) {
        C[row * N + col] = cval;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int64_t N = A.size(0);
    assert(A.dim() == 2 && A.size(0) == A.size(1), "A must be NxN");
    assert(B.dim() == 2 && B.size(0) == B.size(1), "B must be NxN");
    assert(A.size(0) == B.size(0), "A and B must be same size");

    auto C = torch::empty({N, N}, A.options());

    dim3 threadsPerBlock(TS, TS);
    dim3 blocksPerGrid((N + TS - 1) / TS, (N + TS - 1) / TS);

    matmul_symmetric_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        C.data_ptr<float>(),
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        N
    );

    return C;
}
"""

cpp_src = "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"

matmul_cuda_ext = load_inline(
    name="matmul_cuda",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["matmul_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul_cuda = matmul_cuda_ext.matmul_cuda  # Exposed CUDA function

    def forward(self, A, B):
        return self.matmul_cuda(A, B)
