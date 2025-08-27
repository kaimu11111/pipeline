import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mse_kernel(const float* predictions, const float* targets, float* sum, int num_elements) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int block_id = blockIdx.x;
    int block_size = blockDim.x;
    float local_sum = 0.0;

    for (int i = tid + block_id * block_size; i < num_elements; i += gridDim.x * block_size) {
        float diff = predictions[i] - targets[i];
        local_sum += diff * diff;
    }

    shared[tid] = local_sum;
    __syncthreads();

    if (tid == 0) {
        float block_sum = 0;
        for (int i = 0; i < block_size; ++i) {
            block_sum += shared[i];
        }
        atomicAdd(sum, block_sum);
    }
}

torch::Tensor mse_loss_cuda(torch::Tensor predictions, torch::Tensor targets) {
    auto num_elements = predictions.numel();
    auto sum_tensor = torch::empty({1}, predictions.options());
    float* sum_ptr = sum_tensor.data_ptr<float>();

    const int block_size = 256;
    const int grid_size = 128;

    mse_kernel<<<grid_size, block_size, block_size * sizeof(float)>>>(
        predictions.data_ptr<float>(), targets.data_ptr<float>(), sum_ptr, num_elements
    );

    auto mean = sum_tensor.item<float>() / num_elements;
    return torch::full({}, mean, predictions.options());
}
"""

cpp_src = """
torch::Tensor mse_loss_cuda(torch::Tensor predictions, torch::Tensor targets);
"""

mse_loss = load_inline(
    name="mse_loss",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["mse_loss_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.mse_loss = mse_loss

    def forward(self, predictions, targets):
        return self.mse_loss.mse_loss_cuda(predictions, targets)

import time
import torch

def test_mse_loss_cuda(
    device_idx: int = 0,
    N: int = 1_000_000,
    warmup: int = 5,
    repeat: int = 20,
    atol: float = 1e-6,
    rtol: float = 1e-6,
):
    assert torch.cuda.is_available(), "需要 CUDA 环境"
    torch.cuda.set_device(device_idx)

    preds = torch.randn(N, device="cuda", dtype=torch.float32)
    targs = torch.randn(N, device="cuda", dtype=torch.float32)

    # 参考实现
    ref = torch.mean((preds - targs) ** 2)

    model = ModelNew().to("cuda")

    # 预热
    for _ in range(warmup):
        _ = model(preds, targs)
    torch.cuda.synchronize()

    # 数值校验
    out = model(preds, targs)
    torch.cuda.synchronize()
    same = torch.allclose(out, ref, atol=atol, rtol=rtol)
    print(f"[Value] ext={out.item():.8f}  ref={ref.item():.8f}  "
          f"abs_diff={(out-ref).abs().item():.3e}  allclose={same}")

    # 简单基准
    t0 = time.perf_counter()
    for _ in range(repeat):
        _ = model(preds, targs)
    torch.cuda.synchronize()
    print(f"[Perf] N={N:,}  avg={(time.perf_counter()-t0)*1000/repeat:.3f} ms/call")

# 直接调用
if __name__ == "__main__":
    test_mse_loss_cuda(device_idx=0, N=1_000_000)
