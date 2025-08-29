import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

#############################################
# 1. Fused CUDA kernel: element-wise add + ReLU
#############################################
cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void add_relu_kernel(const scalar_t* __restrict__ a,
                                const scalar_t* __restrict__ b,
                                scalar_t* __restrict__ out,
                                int64_t numel) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        scalar_t v = a[idx] + b[idx];
        out[idx] = v > scalar_t(0) ? v : scalar_t(0);
    }
}

torch::Tensor add_relu_cuda(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "Tensors must be on CUDA");
    TORCH_CHECK(a.sizes() == b.sizes(), "Size mismatch");
    TORCH_CHECK(a.scalar_type() == b.scalar_type(), "Dtype mismatch");

    auto out = torch::empty_like(a);
    const int64_t numel = a.numel();
    const int threads = 256;
    const int blocks  = (numel + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.scalar_type(), "add_relu_cuda", ([&](){
        add_relu_kernel<scalar_t><<<blocks, threads>>>(
            a.data_ptr<scalar_t>(),
            b.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            numel);
    }));
    return out;
}
"""

cpp_decl = "torch::Tensor add_relu_cuda(torch::Tensor a, torch::Tensor b);"

fused_ops = load_inline(
    name        = "fused_add_relu",
    cpp_sources = cpp_decl,
    cuda_sources= cuda_src,
    functions   = ["add_relu_cuda"],
    verbose     = False,
)

#############################################
# 2. Network definition using fused kernel
#############################################
class BasicBlockNew(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride     = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # fused element-wise add + ReLU
        out = fused_ops.add_relu_cuda(out.contiguous(), identity.contiguous())
        return out


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.in_channels = 64

        self.conv1   = nn.Conv2d(3, 64, kernel_size=7, stride=2,
                                 padding=3, bias=False)
        self.bn1     = nn.BatchNorm2d(64)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlockNew, 64,  2, stride=1)
        self.layer2 = self._make_layer(BasicBlockNew, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlockNew, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlockNew, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(512 * BasicBlockNew.expansion, num_classes)

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


###############################
# helper for benchmarking
###############################
def get_inputs():
    return [torch.rand(2, 3, 112, 112).cuda()]

def get_init_inputs():
    return []
