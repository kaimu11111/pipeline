cpp
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
    ... 
    dim3 threads(MSE_BLOCK_SIZE, 1, 1); 
    int64_t num_blocks = (n + MSE_BLOCK_SIZE - 1) / MSE_BLOCK_SIZE;
    ... 

    // Secondary reduction to total
    dim3 threads_final(FINAL_BLOCK_SIZE, 1, 1);
    int64_t num_blocks_final = (num_blocks + FINAL_BLOCK_SIZE - 1) / FINAL_BLOCK_SIZE;
    ...
