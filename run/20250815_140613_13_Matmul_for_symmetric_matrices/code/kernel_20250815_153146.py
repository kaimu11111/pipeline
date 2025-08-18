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

    for (int k = 0; k < (N + TS - 1) / TS; k++) {
        // Load A tile
        int a_row = by * TS + ty;
        int a_col = k * TS + tx;
        int a_idx = a_row * N + a_col;
        sA[ty][tx] = (a_row < N && a_col < N) ? A[a_idx] : 0.0f;

        // Load B tile
        int b_row = k * TS + ty;
        int b_col = bx * TS + tx;
        int b_idx = b_row * N + b_col;
        sB[ty][tx] = (b_row < N && b_col < N) ? B[b_idx] : 0.0f;

        __syncthreads();

        // Compute partial products (unrolled loop)
        for (int i = 0; i < TS; i += 4) {
            cval += sA[ty][i] * sB[i][tx];
            if (i + 1 < TS) cval += sA[ty][i+1] * sB[i+1][tx];
            if (i + 2 < TS) cval += sA[ty][i+2] * sB[i+2][tx];
            if (i + 3 < TS) cval += sA[ty][i+3] * sB[i+3][tx];
        }

        __syncthreads();
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
    if (A.dim() != 2 || A.size(0) != A.size(1)) {
        AT_ERROR("A must be NxN");
    }
    if (B.dim() != 2 || B.size(0) != B.size(1)) {
        AT_ERROR("B must be NxN");
    }
    if (A.size(0) != B.size(0)) {
        AT_ERROR("A and B must be same size");
    }
    if (!A.is_contiguous() || !B.is_contiguous()) {
        AT_ERROR("Input tensors must be contiguous");
    }

    auto C = torch::empty({N, N}, A.options());

    dim3 threadsPerBlock(TS, TS);
    dim3 blocksPerGrid((N + TS - 1) / TS, (N + TS - 1) / TS);

    matmul_symmetric_kernel<<<blocksPerGrid, threadsPerBlock>>>(C.data_ptr<float>(), A.data_ptr<float>(), B.data_ptr<float>(), N);
    cudaDeviceSynchronize();

    return C;
}
"""

cpp_src = "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"

matmul_cuda_ext = load_inline(
    name="matmul_cuda",
    cuda_sources=source,
    functions=["matmul_cuda"],
    verbose=True,
    with_cuda=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul_cuda = matmul_cuda_ext.matmul_cuda

    def forward(self, A, B):
        return self.matmul_cuda(A.contiguous(), B.contiguous())
