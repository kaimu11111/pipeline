import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 512

__global__ void compute_mse_sum(const float* predictions, const float* targets, float* partial_sums, int64_t n) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    float val = 0.0f;
    const int idx = bid * blockDim.x + tid;
    if (idx < n) {
        float diff = predictions[idx] - targets[idx];
        val = diff * diff;
    }

    __shared__ float sdata[BLOCK_SIZE];
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

torch::Tensor mse_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    if (predictions.device().type() != torch::kCUDA || targets.device().type() != torch::kCUDA) {
        throw std::invalid_argument("Tensors must be on CUDA device");
    }
    if (predictions.sizes() != targets.sizes()) {
        throw std::invalid_argument("Input tensors must have the same shape");
    }
    if (!predictions.is_contiguous() || !targets.is_contiguous()) {
        predictions = predictions.contiguous();
        targets = targets.contiguous();
    }

    int64_t n = predictions.numel();
    auto options = predictions.options();

    dim3 threads(BLOCK_SIZE, 1, 1);
    int64_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 blocks(num_blocks, 1, 1);

    auto partial_sums = torch::zeros({static_cast<int64_t>(num_blocks)}, options);
    auto partial_ptr = partial_sums.data_ptr<float>();

    compute_mse_sum<<<blocks, threads>>>(predictions.data_ptr<float>(), targets.data_ptr<float>(), partial_ptr, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel configuration error: " + std::string(cudaGetErrorString(err)));
    }

    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel execution error: " + std::string(cudaGetErrorString(sync_err)));
    }

    float total_val = partial_sums.sum().item<float>();
    float mean = total_val / static_cast<float>(n);

    return torch::scalar_tensor(mean, options);
}
"""

cpp_src = "torch::Tensor mse_loss_cuda(torch::Tensor predictions, torch::Tensor targets);"

mse_loss = load_inline(
    name="mse_loss",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["mse_loss_cuda"],
    verbose=False,
    extra_cflags=["-Wno-deprecated-declarations"],
    extra_cuda_cflags=[]
)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss_cuda = mse_loss  # store the module

    def forward(self, predictions, targets):
        return self.mse_loss_cuda.mse_loss_cuda(predictions, targets)
