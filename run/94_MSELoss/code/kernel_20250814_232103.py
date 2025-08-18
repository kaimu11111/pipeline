import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define MSE_BLOCK_SIZE 512
#define FINAL_BLOCK_SIZE 256

__global__ void compute_mse_sum(const float* predictions, const float* targets, float* partial_sums, int64_t n) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    float val = 0.0f;
    const int idx = bid * blockDim.x + tid;
    if (idx < n) {
        float diff = predictions[idx] - targets[idx];
        val = diff * diff;
    }

    __shared__ float sdata[MSE_BLOCK_SIZE];
    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[bid] = sdata[0];
    }
}

__global__ void compute_final_sum(const float* partial_sums, float* total, int64_t m) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    float val = 0.0f;
    const int idx = bid * blockDim.x + tid;
    if (idx < m) {
        val = partial_sums[idx];
    }

    __shared__ float sdata[FINAL_BLOCK_SIZE];
    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(total, sdata[0]);
    }
}

torch::Tensor mse_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    const int64_t n = predictions.numel();
    const int64_t num_blocks = (n + MSE_BLOCK_SIZE - 1) / MSE_BLOCK_SIZE;
    torch::Tensor partial_sums = torch::empty({num_blocks}, torch::kFloat32).to(predictions.device());
    
    dim3 threads_1(MSE_BLOCK_SIZE);
    dim3 blocks_1(num_blocks);
    compute_mse_sum<<<blocks_1, threads_1>>>(predictions.data_ptr<float>(), targets.data_ptr<float>(), partial_sums.data_ptr<float>(), n);
    
    const int64_t num_blocks_final = (num_blocks + FINAL_BLOCK_SIZE - 1) / FINAL_BLOCK_SIZE;
    torch::Tensor total = torch::zeros({1}, torch::kFloat32).to(predictions.device());
    
    dim3 threads_2(FINAL_BLOCK_SIZE);
    dim3 blocks_2(num_blocks_final);
    compute_final_sum<<<blocks_2, threads_2>>>(partial_sums.data_ptr<float>(), total.data_ptr<float>(), num_blocks);
    
    return total / static_cast<float>(n);
}
"""

cpp_src = """
torch::Tensor mse_loss_cuda(torch::Tensor predictions, torch::Tensor targets);
"""

backend = load_inline(name='cuda_mse', 
                     cpp_sources=cpp_src,
                     cuda_sources=source,
                     extra_cuda_cflags=['-arch=sm_70'])

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, predictions, targets):
        return backend.mse_loss_cuda(predictions, targets)
