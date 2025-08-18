import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

#define N 512

__global__ void upper_triangle_mult_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C) {
    int i = blockIdx.x;
    int jdx = threadIdx.x;
    int j = i + jdx;

    if (j >= N) return;

    extern __shared__ float s_A_row[];

    const int N_row_elements = N - i;
    if (threadIdx.x < N_row_elements) {
        int k = i + threadIdx.x;
        int Tk_i = i * (2 * N - i + 1) / 2;
        s_A_row[threadIdx.x] = __ldg(A + Tk_i + (k - i));
    }

    __syncthreads();

    float sum = 0.0f;
    for (int k = i; k <= j; k++) {
        float a_val = s_A_row[k - i];
        int Tk_k = k * (2 * N - k + 1) / 2;
        int bj_off = Tk_k + (j - k);
        float b_val = __ldg(B + bj_off);
        sum = __fma_rn(a_val, b_val, sum);
    }

    // Change the C storage to 2D
    int offset = i * N + j;
    C[offset] = sum;
}

torch::Tensor upper_triangle_mult_cuda(torch::Tensor A, torch::Tensor B) {
    const int block_size = N;
    const int grid_size = N;

    // Changed from packed to full matrix
    auto C = torch::zeros({N, N}, A.options());

    // Check tensors are on CUDA and contiguous
    AT_ASSERTM(A.device().is_cuda(), "Input tensors must be on CUDA");
    AT_ASSERTM(B.device().is_cuda(), "Input tensors must be on CUDA");
    A = A.contiguous();
    B = B.contiguous();

    const size_t shared_mem_size = sizeof(float) * block_size;
    upper_triangle_mult_kernel<<<grid_size, block_size, shared_mem_size>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>()
    );

    cudaDeviceSynchronize();
    return C;
}
"""

cpp_src = "torch::Tensor upper_triangle_mult_cuda(torch::Tensor A, torch::Tensor B);"

upper_triangle_mult = load_inline(
    name="upper_triangle_mult",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["upper_triangle_mult_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.op = upper_triangle_mult

    def forward(self, A, B):
        return self.op.upper_triangle_mult_cuda(A, B)
