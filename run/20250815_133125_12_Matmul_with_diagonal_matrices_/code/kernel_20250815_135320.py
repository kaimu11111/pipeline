# <complete ModelNew code>
import torch
from torch import nn
from torch.utils.cpp_extension import load_inline

source = r"""
__global__ void diag_mult_optimized(const float* A, const float* B, float* C, int N, int M) {
    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x;

    __shared__ float a_val;
    if (tid == 0) {
        a_val = A[row];
    }
    __syncthreads();

    for (int j = tid; j < M; j += blockDim.x) {
        int idx = row * M + j;
        C[idx] = a_val * B[idx];
    }
}
"""

cpp_src = """
#include <torch/types.h>
#include <cuda_runtime.h>

extern "C" __global__ void diag_mult_optimized(const float* A, const float* B, float* C, int N, int M);

void diag_mult_optimizedLauncher(torch::Tensor A, torch::Tensor B, torch::Tensor C, int N, int M) {
    constexpr int THREADS = 256;
    dim3 blocks(N);
    dim3 threads(THREADS);
    diag_mult_optimized<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N, M);
}
"""

cuda_ext = load_inline(
    name='diag_mult_optimized',
    cpp_sources=[cpp_src],
    cuda_sources=[source],
    extra_cuda_cflags=['--expt-extended-lambda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        N = A.size(0)
        M = B.size(1)
        C = torch.empty((N, M), device=A.device, dtype=A.dtype)
        cuda_ext.diag_mult_optimizedLauncher(A, B, C, N, M)
        return C
