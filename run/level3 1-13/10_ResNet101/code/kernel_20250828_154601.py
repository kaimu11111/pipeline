import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
from torch.autograd import Function

# ---------------------------------------------------------------------
# 1. Build the CUDA extension: fused residual addition + ReLU
# ---------------------------------------------------------------------
cuda_src = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_add_relu_kernel(const float* __restrict__ a,
                                      const float* __restrict__ b,
                                      float* __restrict__ out,
                                      const int64_t size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = a[idx] + b[idx];
        out[idx] = val > 0.f ? val : 0.f;
    }
}

torch::Tensor fused_add_relu_cuda(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(a.sizes() == b.sizes(), "Inputs must have the same shape");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(),
                "Inputs must be contiguous");

    auto out = torch::empty_like(a);

    const int64_t numel   = a.numel();
    const int     threads = 256;
    const int     blocks  = (numel + threads - 1) / threads;

    fused_add_relu_kernel<<<blocks, threads>>>(a.data_ptr<float>(),
                                               b.data_ptr<float>(),
                                               out.data_ptr<float>(),
                                               numel);

    return out;
}
'''
cpp_hdr = "torch::Tensor fused_add_relu_cuda(torch::Tensor a, torch::Tensor b);"

fused_add_relu_mod = load_inline(
    name="fused_add_relu_ext",
    cpp_sources=cpp_hdr,
    cuda_sources=cuda_src,
    functions=["fused_add_relu_cuda"],
    verbose=False,
)

# ---------------------------------------------------------------------
# 2. Python-side autograd wrapper
# ---------------------------------------------------------------------
class _FusedAddReLUFn(Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        out = fused_add_relu_mod.fused_add_relu_cuda(a.contiguous(), b.contiguous())
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (out,) = ctx.saved_tensors
        grad = grad_out.clone()
        grad[out <= 0] = 0
        # derivative wrt both inputs is identical
        return grad, grad


class FusedAddReLU(nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return _FusedAddReLUFn.apply(a, b)


# ---------------------------------------------------------------------
# 3. Optimised Bottleneck block using the fused kernel
# ---------------------------------------------------------------------
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_channels * self.expansion)

        self.downsample = downsample
        self.stride     = stride

        # (a + b) -> ReLU fused op
        self.fused_add_relu = FusedAddReLU()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu_(out)  # in-place OK

        out = self.conv2(out)
        out = self.bn2(out)
        out = torch.relu_(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # Fused residual + ReLU
        out = self.fused_add_relu(out, identity)
        return out


# ---------------------------------------------------------------------
# 4. Whole network with fused Bottleneck blocks
# ---------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, layers, num_classes=1000):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1  = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(Bottleneck, 64,  layers[0])
        self.layer2 = self._make_layer(Bottleneck, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(512 * Bottleneck.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ---------------------------------------------------------------------
# 5. Helper functions matching the original interface
# ---------------------------------------------------------------------
batch_size = 5
height = 112
width = 112
layers = [3, 4, 23, 3]
num_classes = 100

def get_inputs():
    return [torch.rand(batch_size, 3, height, width).cuda()]

def get_init_inputs():
    return [layers, num_classes]
