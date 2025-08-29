import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------------
# Inline CUDA kernels: ReLU and 2×2-stride-2 MaxPool
# ---------------------------------------------------------------------------
cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

/* -------------------------  ReLU kernel  ------------------------- */
__global__ void relu_kernel(const float* __restrict__ in,
                            float* __restrict__ out,
                            int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float v = in[idx];
        out[idx] = v > 0.0f ? v : 0.0f;
    }
}

torch::Tensor relu_forward_cuda(torch::Tensor input) {
    const int numel = input.numel();
    auto out = torch::empty_like(input);

    const int threads = 256;
    const int blocks  = (numel + threads - 1) / threads;

    relu_kernel<<<blocks, threads>>>(input.data_ptr<float>(),
                                     out.data_ptr<float>(),
                                     numel);
    return out;
}

/* --------------------- 2×2-stride-2 MaxPool kernel --------------------- */
__global__ void maxpool2x2_kernel(const float* __restrict__ in,
                                  float* __restrict__ out,
                                  int N, int C, int H, int W) {
    /* output tensor shape: (N, C, H/2, W/2)  — row-major (NCHW) */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int H_out = H >> 1;          // H / 2
    int W_out = W >> 1;          // W / 2
    int total = N * C * H_out * W_out;
    if (idx >= total) return;

    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int c     = (idx / (W_out * H_out)) % C;
    int n     = idx / (C * H_out * W_out);

    int h_in = h_out << 1;       // h_out * 2
    int w_in = w_out << 1;       // w_out * 2

    const float* base_in = in + (((n * C + c) * H + h_in) * W + w_in);
    float max_val = base_in[0];
    max_val = fmaxf(max_val, base_in[1]);
    max_val = fmaxf(max_val, base_in[W]);
    max_val = fmaxf(max_val, base_in[W + 1]);

    out[idx] = max_val;
}

torch::Tensor maxpool2x2_forward_cuda(torch::Tensor input) {
    TORCH_CHECK(input.dim() == 4, "Input must be a 4-D tensor (N,C,H,W)");
    int N = input.size(0), C = input.size(1),
        H = input.size(2), W = input.size(3);
    TORCH_CHECK(H % 2 == 0 && W % 2 == 0,
                "Both H and W must be even for 2×2 pooling");

    auto out = torch::empty({N, C, H / 2, W / 2}, input.options());

    int total = N * C * (H / 2) * (W / 2);
    const int threads = 256;
    const int blocks  = (total + threads - 1) / threads;

    maxpool2x2_kernel<<<blocks, threads>>>(input.data_ptr<float>(),
                                           out.data_ptr<float>(),
                                           N, C, H, W);
    return out;
}
"""

cpp_src = """
torch::Tensor relu_forward_cuda(torch::Tensor input);
torch::Tensor maxpool2x2_forward_cuda(torch::Tensor input);
"""

custom_ops = load_inline(
    name="vgg16_fast_ops",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=[
        "relu_forward_cuda",
        "maxpool2x2_forward_cuda",
    ],
    verbose=False,
)

# ---------------------------------------------------------------------------
# Python wrappers for the custom ops
# ---------------------------------------------------------------------------
class FastReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # rely on custom CUDA implementation (out-of-place)
        return custom_ops.relu_forward_cuda(x.contiguous())


class FastMaxPool2d(nn.Module):
    """
    Fixed 2×2 kernel, stride 2, no padding — exactly what VGG uses.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_ops.maxpool2x2_forward_cuda(x.contiguous())

# ---------------------------------------------------------------------------
# Optimised VGG-16 model
# ---------------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        # Replace every ReLU with FastReLU and every MaxPool2d with FastMaxPool2d
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            FastReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            FastReLU(),
            FastMaxPool2d(),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            FastReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            FastReLU(),
            FastMaxPool2d(),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            FastReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            FastReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            FastReLU(),
            FastMaxPool2d(),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            FastReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            FastReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            FastReLU(),
            FastMaxPool2d(),

            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            FastReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            FastReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            FastReLU(),
            FastMaxPool2d(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            FastReLU(),
            nn.Dropout(p=0.0),
            nn.Linear(4096, 4096),
            FastReLU(),
            nn.Dropout(p=0.0),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
