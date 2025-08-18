# <complete ModelNew code>
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define K 256

__global__ void tensor_matrix_mult(
    const float* A,
    const float* B,
    float* C,
    int N, int M, int L
) {
    __shared__ float a_shared[K];

    int n = blockIdx.x / M;
    int m = blockIdx.x % M;
    int l = threadIdx.x;

    // Load A's values into shared memory (coalesced reads)
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        a_shared[i] = A[n * M * K + m * K + i];
    }
    __syncthreads();

    // Compute the dot product with vectorization using 16 floats at a time
    float sum = 0.0f;
    const int stride = 16;
    const int loops = K / stride;
    const float* B_base = (const float*)__builtin_assume_aligned(B + l * K, 16);

    for (int j = 0; j < loops; ++j) {
        int k = j * stride;
        // Load 16 elements at a time for A and B
        float a0 = a_shared[k    ]; float a1 = a_shared[k+1 ]; float a2 = a_shared[k+2 ]; float a3 = a_shared[k+3 ];
        float a4 = a_shared[k+4 ]; float a5 = a_shared[k+5 ]; float a6 = a_shared[k+6 ]; float a7 = a_shared[k+7 ];
        float a8 = a_shared[k+8 ]; float a9 = a_shared[k+9 ]; float a10 = a_shared[k+10]; float a11 = a_shared[k+11];
        float a12 = a_shared[k+12]; float a13 = a_shared[k+13]; float a14 = a_shared[k+14]; float a15 = a_shared[k+15];

        float b0 = B_base[k    ]; float b1 = B_base[k+1 ]; float b2 = B_base[k+2 ]; float b3 = B_base[k+3 ];
        float b4 = B_base[k+4 ]; float b5 = B_base[k+5 ]; float b6 = B_base[k+6 ]; float b7 = B_base[k+7 ];
        float b8 = B_base[k+8 ]; float b9 = B_base[k+9 ]; float b10 = B_base[k+10]; float b11 = B_base[k+11];
        float b12 = B_base[k+12]; float b13 = B_base[k+13]; float b14 = B_base[k+14]; float b15 = B_base[k+15];

        sum += (
            a0*b0 + a1*b1 + a2*b2 + a3*b3 +
            a4*b4 + a5*b5 + a6*b6 + a7*b7 +
            a8*b8 + a9*b9 + a10*b10 + a11*b11 +
            a12*b12 + a13*b13 + a14*b14 + a15*b15
        );
    }

    int pos = n * M * L + m * L + l;
    C[pos] = sum;
}

torch::Tensor tensor_matrix_mult_cuda(torch::Tensor A, torch::Tensor B_t, int N, int M, int L) {
    A = A.contiguous();
    B_t = B_t.contiguous();

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(A.device());
    torch::Tensor C = torch::empty({N, M, L}, options);

    int threads_per_block = L;
    int blocks = N * M;

    tensor_matrix_mult<<<blocks, threads_per_block>>>(
        A.data_ptr<float>(),
        B_t.data_ptr<float>(),
        C.data_ptr<float>(),
        N, M, L
    );
    cudaDeviceSynchronize();

    return C;
}
"""

cpp_src = (
    "torch::Tensor tensor_matrix_mult_cuda(torch::Tensor A, torch::Tensor B_t, int N, int M, int L);"
)

tensor_matrix_mult = load_inline(
    name="tensor_matrix_mult",
    cuda_sources=source,
    cpp_sources=cpp_src,
    functions=["tensor_matrix_mult_cuda"],
    verbose=True,
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3', '-lineinfo', '-Wno-deprecated-gpu-targets'],
    extra_ldflags=[],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.tensor_matrix_mult_cuda = tensor_matrix_mult.tensor_matrix_mult_cuda

    def forward(self, A, B):
        N = A.size(0)
        M = A.size(1)
        K_A = A.size(2)
        L_B = B.size(1)
        assert K_A == 256, "Only valid for K=256 as per problem setup"
        assert A.size(2) == B.size(0), "Incompatible dimensions for A and B"

        B_t = B.t().contiguous()
        return self.tensor_matrix_mult_cuda(A, B_t, N, M, L_B).view(N, M, L_B)
