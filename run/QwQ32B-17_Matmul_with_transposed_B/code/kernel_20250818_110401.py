# <complete ModelNew code>
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TX 16
#define TY 16
#define TW 16

__global__ void matmul_kernel(float* C, const float* A, const float* B, int M, int N, int K) {
    __shared__ float ssa[TX][TW];
    __shared__ float ssb[TW][TY];

    int block_row = blockIdx.x * TX;
    int block_col = blockIdx.y * TY;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float sum = 0.0f;

    for (int it = 0; it < (K + TW - 1) / TW; ++it) {
        int k_start = it * TW;

        // Load A's tile into ssa
        if (block_row + tx < M && k_start + ty < K) {
            ssa[tx][ty] = A[(block_row + tx) * K + (k_start + ty)];
        } else {
            ssa[tx][ty] = 0.0f;
        }

        // Load B's tile into ssb
        int row_B = k_start + tx;
        int col_B = block_col + ty;
        if (row_B < K && col_B < N) {
            ssb[tx][ty] = B[row_B * N + col_B];
        } else {
            ssb[tx][ty] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TW; k += 4) {
            sum += ssa[tx][k] * ssb[k][ty];
            if (k + 1 < TW) sum += ssa[tx][k+1] * ssb[k+1][ty];
            if (k + 2 < TW) sum += ssa[tx][k+2] * ssb[k+2][ty];
            if (k + 3 < TW) sum += ssa[tx][k+3] * ssb[k+3][ty];
        }

        __syncthreads();
    }

    int row = block_row + tx;
    int col = block_col + ty;
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    auto device = A.device();
    assert(B.device() == device);

    int M = A.size(0);
    int K_A = A.size(1);
    int K_B = B.size(0);
    int N_B = B.size(1);
    assert(K_A == K_B);

    auto options = A.options();
    auto C = torch::empty({M, N_B}, options);

    dim3 threads(TX, TY);
    dim3 blocks(
        (M + TX - 1) / TX,
        (N_B + TY - 1) / TY
    );

    matmul_kernel<<<blocks, threads>>>(
        C.data_ptr<float>(),
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        M, N_B, K_B
    );

    return C;
}
"""

cpp_src = "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"

matmul_cuda = load_inline(
    name="matmul_cuda",
    cuda_sources=source,
    cpp_sources=cpp_src,
    functions=["matmul_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return matmul_cuda.matmul_cuda(A.cuda(), B.cuda())
