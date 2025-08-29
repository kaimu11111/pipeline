import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------------
# Hand-written CUDA kernels
# ---------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Kernel 1 : Concatenate two NCHW tensors along channel dimension (dim = 1)
// ---------------------------------------------------------------------------
__global__ void cat_channels_kernel(const float* __restrict__ a,
                                    const float* __restrict__ b,
                                    float* __restrict__ out,
                                    int N, int C1, int C2,
                                    int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * (C1 + C2) * H * W;
    if (idx >= total) return;

    int inner = H * W;
    int n   = idx / ((C1 + C2) * inner);
    int rem = idx % ((C1 + C2) * inner);
    int c   = rem / inner;
    int hw  = rem % inner;

    if (c < C1) {
        int a_idx = n * C1 * inner + c * inner + hw;
        out[idx] = a[a_idx];
    } else {
        int c_b  = c - C1;
        int b_idx = n * C2 * inner + c_b * inner + hw;
        out[idx] = b[b_idx];
    }
}

torch::Tensor cat_channels_cuda(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(a.dtype() == torch::kFloat32 && b.dtype() == torch::kFloat32,
                "Only float32 tensors are supported");
    TORCH_CHECK(a.dim() == 4 && b.dim() == 4, "Expect 4-D NCHW tensors");
    TORCH_CHECK(a.size(0) == b.size(0) && a.size(2) == b.size(2) &&
                a.size(3) == b.size(3),
                "Batch, height, and width dimensions must match");

    int N = a.size(0);
    int C1 = a.size(1);
    int C2 = b.size(1);
    int H  = a.size(2);
    int W  = a.size(3);

    auto out = torch::empty({N, C1 + C2, H, W}, a.options());

    int total = out.numel();
    int threads = 256;
    int blocks  = (total + threads - 1) / threads;

    cat_channels_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C1, C2, H, W);

    return out;
}

// ---------------------------------------------------------------------------
// Kernel 2 : Global average pooling (adaptive_avg_pool2d to 1×1)
//            Input  N×C×H×W  →  Output  N×C
// ---------------------------------------------------------------------------
__global__ void global_avg_pool2d_kernel(const float* __restrict__ inp,
                                         float* __restrict__ out,
                                         int N, int C, int H, int W) {
    int tid   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C;
    if (tid >= total) return;

    int n = tid / C;
    int c = tid % C;

    int stride = H * W;
    int base   = n * C * stride + c * stride;

    float sum = 0.0f;
    for (int i = 0; i < stride; ++i) {
        sum += inp[base + i];
    }
    out[tid] = sum / static_cast<float>(stride);
}

torch::Tensor global_avg_pool2d_cuda(torch::Tensor inp) {
    TORCH_CHECK(inp.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(inp.dtype() == torch::kFloat32, "Only float32 tensors supported");
    TORCH_CHECK(inp.dim() == 4, "Expect a 4-D NCHW tensor");

    int N = inp.size(0);
    int C = inp.size(1);
    int H = inp.size(2);
    int W = inp.size(3);

    auto out = torch::empty({N, C}, inp.options());

    int total   = N * C;
    int threads = 256;
    int blocks  = (total + threads - 1) / threads;

    global_avg_pool2d_kernel<<<blocks, threads>>>(
        inp.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, H, W);

    return out;
}
"""

cpp_source = r"""
torch::Tensor cat_channels_cuda(torch::Tensor a, torch::Tensor b);
torch::Tensor global_avg_pool2d_cuda(torch::Tensor inp);
"""

# Compile and load the CUDA extension
_fast_ops = load_inline(
    name="fast_ops",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["cat_channels_cuda", "global_avg_pool2d_cuda"],
    verbose=False,
)

# ---------------------------------------------------------------------------
# PyTorch modules using the custom kernels
# ---------------------------------------------------------------------------
class DenseBlock(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(self._make_layer(num_input_features + i * growth_rate,
                                           growth_rate))
        self.layers = nn.ModuleList(layers)

    @staticmethod
    def _make_layer(in_features: int, growth_rate: int):
        # Standard BN → ReLU → Conv layout
        return nn.Sequential(
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1,
                      bias=False),
            nn.Dropout(0.0)
        )

    def forward(self, x):
        # Iteratively concatenate using hand-written CUDA kernel
        for layer in self.layers:
            new_feature = layer(x)
            # Ensure contiguous memory before passing to CUDA kernel
            x = _fast_ops.cat_channels_cuda(x.contiguous(),
                                            new_feature.contiguous())
        return x


class TransitionLayer(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        super().__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features, num_output_features,
                      kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.transition(x)


class ModelNew(nn.Module):
    def __init__(self, growth_rate: int = 32, num_classes: int = 1000):
        super().__init__()
        self._fast_ops = _fast_ops  # expose to forward if needed

        # Initial stem
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Dense blocks + transition layers
        num_features = 64
        block_layers = [6, 12, 48, 32]   # DenseNet-201 layout

        self.dense_blocks      = nn.ModuleList()
        self.transition_layers = nn.ModuleList()

        for i, num_layers in enumerate(block_layers):
            block = DenseBlock(num_layers=num_layers,
                               num_input_features=num_features,
                               growth_rate=growth_rate)
            self.dense_blocks.append(block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_layers) - 1:  # no transition after the last block
                trans = TransitionLayer(num_input_features=num_features,
                                        num_output_features=num_features // 2)
                self.transition_layers.append(trans)
                num_features = num_features // 2

        # Final BN and classifier
        self.final_bn  = nn.BatchNorm2d(num_features)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)

        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i != len(self.dense_blocks) - 1:
                x = self.transition_layers[i](x)

        x = self.final_bn(x)
        x = F.relu(x, inplace=True)

        # Replace adaptive_avg_pool2d with custom CUDA global avg pool
        x = self._fast_ops.global_avg_pool2d_cuda(x)  # shape: N × C

        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# Helper functions for external runners
# ---------------------------------------------------------------------------
batch_size = 5
num_classes = 5
height, width = 112, 112

def get_inputs():
    return [torch.rand(batch_size, 3, height, width, device="cuda")]

def get_init_inputs():
    return [32, num_classes]
