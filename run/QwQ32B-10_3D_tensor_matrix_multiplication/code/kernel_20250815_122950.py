# <your corrected code>
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

    // Load A into shared memory
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        a_shared[i] = A[n * M * K + m * K + i];
    }
    __syncthreads();

    // Process all l elements in grid-stride loop
    for (int l = threadIdx.x; l < L; l += blockDim.x) {
        float sum = 0.0f;
        const float* B_base = reinterpret_cast<const float*>(__builtin_assume_aligned(B + l * K, 16));

        const int stride = 16;
        const int loops = K / stride;

        for (int j = 0; j < loops; ++j) {
            int k = j * stride;
            sum = fmaf(a_shared[k    ], __ldg(B_base + k    ), sum);
            sum = fmaf(a_shared[k + 1], __ldg(B_base + k + 1), sum);
            sum = fmaf(a_shared[k + 2], __ldg(B_base + k + 2), sum);
            sum = fmaf(a_shared[k + 3], __ldg(B_base + k + 3), sum);
            sum = fmaf(a_shared[k + 4], __ldg(B_base + k + 4), sum);
            sum = fmaf(a_shared[k + 5], __ldg(B_base + k + 5), sum);
            sum = fmaf(a_shared[k + 6], __ldg(B_base + k + 6), sum);
            sum = fmaf(a_shared[k + 7], __ldg(B_base + k + 7), sum);
            sum = fmaf(a_shared[k + 8], __ldg(B_base + k + 8), sum);
            sum = fmaf(a_shared[k + 9], __ldg(B_base + k + 9), sum);
            sum = fmaf(a_shared[k +10], __ldg(B_base + k +10), sum);
            sum = fmaf(a_shared[k +11], __ldg(B_base + k +11), sum);
            sum = fmaf(a_shared[k +12], __ldg(B_base + k +12), sum);
            sum = fmaf(a_shared[k +13], __ldg(B_base + k +13), sum);
            sum = fmaf(a_shared[k +14], __ldg(B_base + k +14), sum);
            sum = fmaf(a_shared[k +15], __ldg(B_base + k +15), sum);
        }

        int pos = n * M * L + m * L + l;
        C[pos] = sum;
    }
}

torch::Tensor tensor_matrix_mult_cuda(torch::Tensor A, torch::Tensor B_t, int N, int M, int L) {
    A = A.contiguous();
    B_t = B_t.contiguous();

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(A.device());
    torch::Tensor C = torch::empty({N, M, L}, options);

    int threads_per_block = 64;  // Optimized for occupancy and SM utilization
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

cpp_src = "torch::Tensor tensor_matrix_mult_cuda(torch::Tensor A, torch::Tensor B_t, int N, int M, int L);"

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
