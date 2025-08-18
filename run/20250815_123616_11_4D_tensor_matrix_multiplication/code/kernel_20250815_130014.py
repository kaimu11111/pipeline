import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void tensor_matmul_kernel(const float* A, const float* B, float* C, int R, int L, int K) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx >= R * K) return;

    int r = global_idx / K;
    int k = global_idx % K;

    float sum = 0.f;
    for (int l = 0; l < L; ++l) {
        const int a_idx = r * L + l;
        const int b_idx = l * K + k;
        sum += A[a_idx] * B[b_idx];
    }
    const int c_idx = r * K + k;
    C[c_idx] = sum;
}

torch::Tensor tensor_matmul_cuda(
    const torch::Tensor &A,
    const torch::Tensor &B
) {
    const int R = A.size(0) * A.size(1) * A.size(2);
    const int L = A.size(3);
    const int K = B.size(1);

    auto C = torch::empty({R, K}, A.options());
    const int threads_per_block = 256;
    const int numel = R * K;
    const int blocks = (numel + threads_per_block - 1) / threads_per_block;

    tensor_matmul_kernel<<<blocks, threads_per_block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        R, L, K
    );
    return C.view({A.size(0), A.size(1), A.size(2), K});
}
"""

cpp_src = """
#include <torch/extension.h>
#include <pybind11/pybind11.h>

at::Tensor tensor_matmul_cuda(const at::Tensor &A, const at::Tensor &B);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tensor_matmul_cuda", &tensor_matmul_cuda, "Tensor matmul CUDA");
}
"""

tensor_matmul = load_inline(
    name="tensor_matmul",
    cpp_sources=cpp_src,
    cuda_sources=source,
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.tensor_matmul_cuda = tensor_matmul.tensor_matmul_cuda

    def forward(self, A, B):
        A = A.contiguous()
        B = B.contiguous()
        return self.tensor_matmul_cuda(A, B)
