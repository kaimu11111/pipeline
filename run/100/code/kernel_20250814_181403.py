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
    int D
) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int i = bid * blockDim.x + tid;
    if (i >= B * D) return;
    int j = i / D;
    int k = i % D;
    float p = predictions[j * D + k];
    float target = targets[j];
    float val = 1.0f - p * target;
    float contrib = (val > 0.0f) ? val : 0.0f;
    sdata[tid] = contrib;
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

__global__ void compute_hinge_loss_part2(
    const float* partial_sums,
    float* total,
    int M
) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    float local_sum = 0.0f;
    for (int idx = tid; idx < M; idx += blockDim.x) {
        local_sum += partial_sums[idx];
    }
    sdata[tid] = local_sum;
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

torch::Tensor hinge_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    auto B = predictions.size(0);
    auto D = predictions.size(1);
    auto N = B * D;
    const int block_size_part1 = 256;
    const int grid_size_part1 = (N + block_size_part1 - 1) / block_size_part1;
    const int M = grid_size_part1;

    auto partial_sums = torch::empty({M}, predictions.options());
    auto total = torch::zeros(1, predictions.options());

    dim3 blocks_part1(grid_size_part1);
    dim3 threads_part1(block_size_part1);

    compute_hinge_loss_part1<<<blocks_part1, threads_part1, block_size_part1 * sizeof(float)>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        partial_sums.data_ptr<float>(),
        B, D
    );

    const int block_size_part2 = 1024;
    dim3 blocks_part2(1);
    dim3 threads_part2(block_size_part2);

    compute_hinge_loss_part2<<<blocks_part2, threads_part2, block_size_part2 * sizeof(float)>>>(
        partial_sums.data_ptr<float>(),
        total.data_ptr<float>(),
        M
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
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.hinge_loss = hinge_loss

    def forward(self, predictions, targets):
        return self.hinge_loss.hinge_loss_cuda(predictions, targets)
