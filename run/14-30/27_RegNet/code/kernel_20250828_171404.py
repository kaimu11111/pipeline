import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------------
# CUDA source: fused Global Average Pooling and ReLU
# ------------------------------------------------------------------
source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// ---------------------------------------------
// Element-wise ReLU
// ---------------------------------------------
__global__ void relu_kernel(const float* __restrict__ input,
                            float* __restrict__ output,
                            const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float v = input[idx];
        output[idx] = v > 0.f ? v : 0.f;
    }
}

torch::Tensor relu_forward_cuda(torch::Tensor input) {
    CHECK_INPUT(input);
    auto output = torch::empty_like(input);
    const int size    = input.numel();
    const int threads = 256;
    const int blocks  = (size + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(input.data_ptr<float>(),
                                     output.data_ptr<float>(),
                                     size);
    return output;
}

// ---------------------------------------------
// Global Average Pooling (N, C, H, W) → (N, C)
// ---------------------------------------------
__global__ void gap_kernel(const float* __restrict__ input,
                           float* __restrict__ output,
                           const int N, const int C,
                           const int H, const int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int NC  = N * C;
    if (idx < NC) {
        int n = idx / C;
        int c = idx % C;
        const float* in_ptr = input + ((n * C + c) * H * W);
        float sum = 0.f;
        for (int i = 0; i < H * W; ++i) {
            sum += in_ptr[i];
        }
        output[n * C + c] = sum / (H * W);
    }
}

torch::Tensor global_avg_pool_forward_cuda(torch::Tensor input) {
    CHECK_INPUT(input);
    TORCH_CHECK(input.dim() == 4, "Input must be 4-D (N, C, H, W)");
    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);

    auto output = torch::empty({N, C}, input.options());
    const int threads = 256;
    const int blocks  = (N * C + threads - 1) / threads;

    gap_kernel<<<blocks, threads>>>(input.data_ptr<float>(),
                                    output.data_ptr<float>(),
                                    N, C, H, W);
    return output;
}
"""

# ------------------------------------------------------------------
# C++ prototypes
# ------------------------------------------------------------------
cpp_src = """
torch::Tensor relu_forward_cuda(torch::Tensor input);
torch::Tensor global_avg_pool_forward_cuda(torch::Tensor input);
"""

# ------------------------------------------------------------------
# Compile / load
# ------------------------------------------------------------------
custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["relu_forward_cuda", "global_avg_pool_forward_cuda"],
    verbose=False,
)

# ------------------------------------------------------------------
# Optimised model definition
# ------------------------------------------------------------------
class ModelNew(nn.Module):
    # --------------------------------------------------------------
    # Nested wrappers for custom kernels
    # --------------------------------------------------------------
    class CustomReLU(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return custom_ops.relu_forward_cuda(x.contiguous())

    def __init__(self, input_channels, stages, block_widths, output_classes):
        super().__init__()
        self.stages        = stages
        self.block_widths  = block_widths

        layers = []
        current_channels = input_channels
        for i in range(stages):
            layers.append(self._make_stage(current_channels, block_widths[i]))
            current_channels = block_widths[i]

        self.feature_extractor = nn.Sequential(*layers)
        self.fc = nn.Linear(block_widths[-1], output_classes)

    # --------------------------------------------------------------
    # Helper stage builder
    # --------------------------------------------------------------
    def _make_stage(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=True),   # fixed bias
            nn.BatchNorm2d(out_channels),
            ModelNew.CustomReLU(),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=True),   # fixed bias
            nn.BatchNorm2d(out_channels),
            ModelNew.CustomReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    # --------------------------------------------------------------
    # Forward
    # --------------------------------------------------------------
    def forward(self, x):
        x = self.feature_extractor(x)
        x = custom_ops.global_avg_pool_forward_cuda(x.contiguous())
        x = self.fc(x)
        return x
