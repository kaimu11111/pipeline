# <optimized code>
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void compute_hinge_loss_part1(
    const float* predictions,
    const float* targets,
    float* partial_sums,
    int B,
    int D,
    int R
) {
    extern __shared__ float shared_mem[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int i = bid * blockDim.x + tid;
    if (i >= B * D) return;
    
    float *sdata = shared_mem;
    float *shared_targets = &shared_mem[blockDim.x];
    
    int j_start = (bid * blockDim.x) / D;
    if (tid < R) {
        shared_targets[tid] = targets[j_start + tid];
    }
    __syncthreads();
    
    int j = i / D;
    int local_row = j - j_start;
    float target = shared_targets[local_row];
    sdata[tid] = fmaxf(0.0f, 1.0f - predictions[i] * target);
    
    __syncthreads();
    
    // Warp-based reduction within shared_mem
    float sum_val = sdata[tid];
    for (int d = 16; d > 0; d >>= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFF, sum_val, d, 32);
        sum_val += other;
    }
    
    if ((tid % 32) == 0) {
        int warpID = tid / 32;
        sdata[warpID] = sum_val;
    }
    __syncthreads();
    
    // Final reduction of 32 warp contributions via shuffle
    if (tid >= blockDim.x / 32) return;
    
    float total = sdata[tid];
    for (int d = 16; d > 0; d >>= 1) {
        float temp = __shfl_xor_sync(0xFFFFFFFF, total, d, 32);
        total += temp;
    }
    
    if (tid == 0) {
        partial_sums[bid] = total;
    }
}

__global__ void compute_hinge_loss_part2(
    const float* partial_sums,
    float* total,
    int M
) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int num_warps = block_size /32;
    
    int warp_id = tid /32;
    int lane = tid %32;
    
    int chunk_size = (M + num_warps -1)/ num_warps;
    int start = warp_id * chunk_size;
    int end = min(start + chunk_size, M);
    
    float local_sum = 0;
    for (int i = start + lane; i < end; i +=32) {
        local_sum += partial_sums[i];
    }

    // Reduce within warp
    for (int d =16; d>0; d>>=1) {
        float temp = __shfl_xor_sync(0xFFFFFFFF, local_sum, d,32);
        local_sum += temp;
    }
    
    if (lane ==0) {
        sdata[warp_id] = local_sum;
    }
    __syncthreads();
    
    // Reduce 32 warp sums using first warp
    if (tid < num_warps) {
        float sum_total = sdata[tid];
        for (int d =16; d>0; d>>=1) {
            float temp = __shfl_xor_sync(0xFFFFFFFF, sum_total, d,32);
            sum_total += temp;
        }
        if (tid ==0) {
            atomicAdd(total, sum_total);
        }
    }
}

torch::Tensor hinge_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    auto B = predictions.size(0);
    auto D = predictions.size(1);
    int N = B * D;
    const int block_size_part1 = 1024;
    const int grid_size_part1 = (N + block_size_part1 - 1) / block_size_part1;
    const int R = (block_size_part1 + D -1) / D;
    const int sharedMemSize_part1 = (block_size_part1 + R) * sizeof(float);
    
    auto partial_sums = torch::empty({grid_size_part1}, predictions.options());
    auto total = torch::zeros(1, predictions.options());
    
    dim3 blocks_part1(grid_size_part1);
    dim3 threads_part1(block_size_part1);
    
    compute_hinge_loss_part1<<<blocks_part1, threads_part1, sharedMemSize_part1>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        partial_sums.data_ptr<float>(),
        B, D, R
    );
    
    const int block_size_part2 = 1024;
    const int sharedMemSize_part2 = block_size_part2 * sizeof(float);
    dim3 blocks_part2(1);
    dim3 threads_part2(block_size_part2);
    
    compute_hinge_loss_part2<<<blocks_part2, threads_part2, sharedMemSize_part2>>>(
        partial_sums.data_ptr<float>(),
        total.data_ptr<float>(),
        grid_size_part1
    );
    
    float mean = total[0].item<float>() / static_cast<float>(N);
    return torch::tensor({mean}, predictions.options());
}
"""

cpp_src = """
torch::Tensor hinge_loss_cuda(torch::Tensor predictions, torch::Tensor targets);
"""

hinge_loss = load_inline(
    name="hinge_loss",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["hinge_loss_cuda"],
    verbose=True,
    extra_cuda_cflags=["-O3", "-maxrregcount=32"]
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.hinge_loss = hinge_loss

    def forward(self, predictions, targets):
        return self.hinge_loss.hinge_loss_cuda(predictions, targets)
