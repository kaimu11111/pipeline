import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void diag_mult_kernel(const float* A, const float* B, float* C, int N, int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * M) {
        int i = idx / M;
        int j = idx % M;
        C[idx] = A[i] * B[idx];
    }
}

torch::Tensor diag_mult_cuda(torch::Tensor A, torch::Tensor B) {
    int64_t N = A.size(0);
    int64_t M = B.size(1);
    auto C = torch::empty_like(B);

    const int threads_per_block = 256;
    const int blocks_per_grid = (N * M + threads_per_block - 1) / threads_per_block;

    diag_mult_kernel<<<blocks_per_grid, threads_per_block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N, M
    );

    return C;
}
"""

cpp_src = """
torch::Tensor diag_mult_cuda(torch::Tensor A, torch::Tensor B);
"""

diag_mult = load_inline(
    name="diag_mult",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["diag_mult_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.diag_mult = diag_mult

    def forward(self, A, B):
        return self.diag_mult.diag_mult_cuda(A, B)
