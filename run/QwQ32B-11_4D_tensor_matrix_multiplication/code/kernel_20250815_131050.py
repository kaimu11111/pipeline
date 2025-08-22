import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_DIM 16  // Tile size per block (16x16)
#define TILE_L 32    // Chunk size of L dimension

__global__ void tensor_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int R,
    int L,
    int K
) {
    const int block_row = blockIdx.x * TILE_DIM;
    const int block_col = blockIdx.y * TILE_DIM;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row = block_row + tx;
    const int col = block_col + ty;

    float sum = 0.0f;

    __shared__ float shared_A[TILE_DIM][TILE_L];  // Rows from A for current block + current L chunk
    __shared__ float shared_B[TILE_L][TILE_DIM];  // Columns from B for current block + current L chunk

    for (int l = 0; l < L; l += TILE_L) {
        const int l_start = l;
        const int l_end = min(l_start + TILE_L, L);

        // Load A's data into shared memory
        for (int e = ty; e < (l_end - l_start); e += blockDim.y) {
            const int l_rel = e;
            const int global_l = l_start + l_rel;
            shared_A[tx][l_rel] = (row < R) ? A[row * L + global_l] : 0.0f;
        }

        // Load B's data into shared memory
        for (int e = tx; e < (l_end - l_start); e += blockDim.x) {
            const int l_rel = e;
            const int global_col = block_col + ty;
            shared_B[l_rel][ty] = (global_col < K) ? B[(l_start + l_rel) * K + global_col] : 0.0f;
        }

        __syncthreads();

        // Compute partial sum using the tile in shared memory
        for (int i = 0; i < (l_end - l_start); ++i) {
            const float a_val = shared_A[tx][i];
            const float b_val = shared_B[i][ty];
            sum += a_val * b_val;
        }

        __syncthreads();
    }

    // Write result if within bounds
    if (row < R && col < K) {
        const int c_idx = row * K + col;
        C[c_idx] = sum;
    }
}

torch::Tensor tensor_matmul_cuda(
    const torch::Tensor &A,
    const torch::Tensor &B
) {
    const int R = A.size(0) * A.size(1) * A.size(2);
    const int L = A.size(3);
    const int K = B.size(1);

    auto C = torch::empty({R, K}, A.options());

    dim3 threadsPerBlock(TILE_DIM, TILE_DIM);  // 16x16 block
    int blocks_x = (R + TILE_DIM - 1) / TILE_DIM;
    int blocks_y = (K + TILE_DIM - 1) / TILE_DIM;
    dim3 blocks(blocks_x, blocks_y);

    tensor_matmul_kernel<<<blocks, threadsPerBlock>>>(
        A.contiguous().data_ptr<float>(),
        B.contiguous().data_ptr<float>(),
        C.data_ptr<float>(),
        R, L, K
    );

    return C.view({A.size(0), A.size(1), A.size(2), K});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tensor_matmul_cuda", &tensor_matmul_cuda, "Optimized tensor matmul CUDA kernel");
}

source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_DIM 16
#define TILE_L 32

__global__ void tensor_matmul_kernel(...) { /* Same as above */ }

// Implementation as above
"""

cpp_src = """
#include <torch/extension.h>
#include <pybind11/pybind11.h>

at::Tensor tensor_matmul_cuda(const at::Tensor &A, const at::Tensor &B);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tensor_matmul_cuda", &tensor_matmul_cuda, "Optimized tensor matmul CUDA kernel");
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
