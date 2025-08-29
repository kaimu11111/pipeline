import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# ----------------------------------------------------------------------
# CUDA Kernels (Channel Shuffle + Residual Add)
# ----------------------------------------------------------------------
cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// ------------------------ Channel Shuffle ----------------------------
__global__ void channel_shuffle_kernel(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       const int batch,
                                       const int channels,
                                       const int height,
                                       const int width,
                                       const int groups) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch * channels * height * width;
    if (idx >= total) return;

    int w      =  idx % width;
    int tmp1   =  idx / width;
    int h      =  tmp1 % height;
    int tmp2   =  tmp1 / height;
    int c_out  =  tmp2 % channels;
    int n      =  tmp2 / channels;

    const int channels_per_group = channels / groups;
    const int g = c_out % groups;
    const int c = c_out / groups;

    const int c_in = g * channels_per_group + c;

    const int input_idx = (((n * channels + c_in) * height + h) * width + w);
    output[idx] = input[input_idx];
}

// ------------------------- Residual Add ------------------------------
__global__ void residual_add_kernel(const float* __restrict__ a,
                                    const float* __restrict__ b,
                                    float* __restrict__ out,
                                    const int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] + b[idx];
}

// ------------------------- C++ Bindings ------------------------------
torch::Tensor channel_shuffle_cuda(torch::Tensor input, int64_t groups) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    const int batch    = input.size(0);
    const int channels = input.size(1);
    const int height   = input.size(2);
    const int width    = input.size(3);

    auto output = torch::empty_like(input);

    const int total = batch * channels * height * width;
    const int block = 256;
    const int grid  = (total + block - 1) / block;

    channel_shuffle_kernel<<<grid, block>>>(input.data_ptr<float>(),
                                            output.data_ptr<float>(),
                                            batch, channels, height, width,
                                            groups);
    return output;
}

torch::Tensor residual_add_cuda(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(a.sizes() == b.sizes(), "size mismatch for residual add");
    auto out = torch::empty_like(a);
    const int n     = a.numel();
    const int block = 256;
    const int grid  = (n + block - 1) / block;
    residual_add_kernel<<<grid, block>>>(a.data_ptr<float>(),
                                         b.data_ptr<float>(),
                                         out.data_ptr<float>(),
                                         n);
    return out;
}
"""

cpp_src = r"""
torch::Tensor channel_shuffle_cuda(torch::Tensor input, int64_t groups);
torch::Tensor residual_add_cuda(torch::Tensor a, torch::Tensor b);
"""

cuda_ops = load_inline(
    name="shufflenet_cuda_ops",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["channel_shuffle_cuda", "residual_add_cuda"],
    verbose=False,
)

# ----------------------------------------------------------------------
# Autograd-compatible wrappers
# ----------------------------------------------------------------------
class ChannelShuffleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, groups: int):
        ctx.groups = groups
        return cuda_ops.channel_shuffle_cuda(x, groups)

    @staticmethod
    def backward(ctx, grad_out):
        # Channel shuffle is its own inverse permutation
        grad_in = cuda_ops.channel_shuffle_cuda(grad_out.contiguous(), ctx.groups)
        return grad_in, None


class ResidualAddFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        return cuda_ops.residual_add_cuda(a, b)

    @staticmethod
    def backward(ctx, grad_out):
        # dL/da = grad_out ; dL/db = grad_out
        return grad_out.clone(), grad_out.clone()


def residual_add(a, b):
    return ResidualAddFunction.apply(a, b)


class ChannelShuffleCUDA(nn.Module):
    def __init__(self, groups: int):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        return ChannelShuffleFunction.apply(x, self.groups)

# ----------------------------------------------------------------------
# Optimised Model Definition
# ----------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3):
        super().__init__()
        assert out_channels % 4 == 0
        mid_channels = out_channels // 4

        # 1×1 group convolution
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False, groups=groups)
        self.bn1   = nn.BatchNorm2d(mid_channels)

        # 3×3 depth-wise convolution
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1,
                               bias=False, groups=mid_channels)
        self.bn2   = nn.BatchNorm2d(mid_channels)

        # 1×1 group convolution
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False, groups=groups)
        self.bn3   = nn.BatchNorm2d(out_channels)

        # Custom CUDA channel shuffle
        self.shuffle = ChannelShuffleCUDA(groups)

        # Shortcut
        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.shuffle(out)
        out = F.relu(self.bn3(self.conv3(out)))
        out = residual_add(out, self.shortcut(x))
        return out

# ----------------------------------------------------------------------
# Helper functions (unchanged interface)
# ----------------------------------------------------------------------
batch_size      = 5
input_channels  = 120
out_channels    = 240
groups          = 3
height, width   = 224, 224

def get_inputs():
    return [torch.rand(batch_size, input_channels, height, width, device="cuda")]

def get_init_inputs():
    return [input_channels, out_channels, groups]
