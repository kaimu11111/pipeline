# 1. Imports ──────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# 2. CUDA source ──────────────────────────────────────────────────────
source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>      // ← gives current CUDA stream
#include <cuda_runtime.h>

////////////////////////////////////////////////////////////////
// fused ReLU + 2×2 max-pool kernel
////////////////////////////////////////////////////////////////
template <typename scalar_t>
__global__ void relu_maxpool2x2_kernel(const scalar_t* __restrict__ in,
                                       scalar_t*       __restrict__ out,
                                       const int N, const int C,
                                       const int H, const int W)
{
    const int H_out = H >> 1;      // H / 2
    const int W_out = W >> 1;      // W / 2
    const int total = N * C * H_out * W_out;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // Decompose linear index → n, c, h_out, w_out
    int w_out = idx % W_out;   idx /= W_out;
    int h_out = idx % H_out;   idx /= H_out;
    int c     = idx % C;       idx /= C;
    int n     = idx;

    const int h_in = h_out << 1;   // *2
    const int w_in = w_out << 1;

    const int in_off = (((n * C + c) * H + h_in) * W) + w_in;

    scalar_t v0 = in[in_off];
    scalar_t v1 = in[in_off + 1];
    scalar_t v2 = in[in_off + W];
    scalar_t v3 = in[in_off + W + 1];

    scalar_t m1 = v0 > v1 ? v0 : v1;
    scalar_t m2 = v2 > v3 ? v2 : v3;
    scalar_t mx = m1 > m2 ? m1 : m2;
    mx = mx > scalar_t(0) ? mx : scalar_t(0);   // ReLU

    out[(((n * C + c) * H_out + h_out) * W_out) + w_out] = mx;
}

////////////////////////////////////////////////////////////////
// host launcher
////////////////////////////////////////////////////////////////
torch::Tensor relu_maxpool2x2_cuda(torch::Tensor input)
{
    TORCH_CHECK(input.is_cuda(),  "input must reside on CUDA");
    TORCH_CHECK(input.dim() == 4, "input must be 4-D (NCHW)");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32,
                "only float32 supported");
    const int64_t N = input.size(0);
    const int64_t C = input.size(1);
    const int64_t H = input.size(2);
    const int64_t W = input.size(3);
    TORCH_CHECK((H & 1) == 0 && (W & 1) == 0,
                "H and W must be even to 2×2-pool");

    auto output = torch::empty({N, C, H / 2, W / 2}, input.options());

    const int threads = 256;
    const int64_t total = N * C * (H / 2) * (W / 2);
    const int blocks = (total + threads - 1) / threads;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    relu_maxpool2x2_kernel<float>
        <<<blocks, threads, 0, stream>>>(input.data_ptr<float>(),
                                         output.data_ptr<float>(),
                                         static_cast<int>(N),
                                         static_cast<int>(C),
                                         static_cast<int>(H),
                                         static_cast<int>(W));
    TORCH_CUDA_CHECK(cudaGetLastError());
    return output;
}
"""

# 3. C++ prototypes ───────────────────────────────────────────────────
cpp_src = "torch::Tensor relu_maxpool2x2_cuda(torch::Tensor input);"

# 4. load_inline call ────────────────────────────────────────────────
fused_relu_pool = load_inline(
    name="fused_relu_maxpool2x2",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["relu_maxpool2x2_cuda"],
    verbose=False,
)

# 5. PyTorch module ──────────────────────────────────────────────────
class ModelNew(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1,  6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        self.relu_pool = fused_relu_pool.relu_maxpool2x2_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu_pool(x)

        x = self.conv2(x)
        x = self.relu_pool(x)

        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
