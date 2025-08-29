import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------
# CUDA kernel + host wrapper
# ---------------------------------------------------------------------
source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void softmax_lastdim_kernel(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       int rows,
                                       int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    const float* row_in  = input  + row * cols;
    float*       row_out = output + row * cols;

    // 1. find max
    float max_val = row_in[0];
    for (int i = 1; i < cols; ++i)
        max_val = fmaxf(max_val, row_in[i]);

    // 2. exponentiate & accumulate sum
    float sum_exp = 0.f;
    for (int i = 0; i < cols; ++i) {
        float e = expf(row_in[i] - max_val);
        row_out[i] = e;           // store temporary exp result
        sum_exp += e;
    }

    // 3. normalize
    float inv_sum = 1.f / sum_exp;
    for (int i = 0; i < cols; ++i)
        row_out[i] *= inv_sum;
}

torch::Tensor softmax_lastdim_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must reside on CUDA device");

    auto input_contig = input.contiguous();
    auto output       = torch::empty_like(input_contig);

    const int64_t rows = input_contig.numel() / input_contig.size(-1);
    const int64_t cols = input_contig.size(-1);

    const int block_size = 256;
    const int grid_size  = (rows + block_size - 1) / block_size;

    softmax_lastdim_kernel<<<grid_size, block_size>>>(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(rows),
        static_cast<int>(cols)
    );

    return output;
}
"""

# ---------------------------------------------------------------------
# C++ forward declaration(s) for functions exposed to Python
# ---------------------------------------------------------------------
cpp_src = r"""
#include <torch/extension.h>
torch::Tensor softmax_lastdim_cuda(torch::Tensor input);
"""

# ---------------------------------------------------------------------
# Build / load the CUDA extension
# ---------------------------------------------------------------------
_softmax_ext = load_inline(
    name="fast_softmax_lastdim",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["softmax_lastdim_cuda"],
    verbose=False
)

# ---------------------------------------------------------------------
# Python-side convenience wrappers
# ---------------------------------------------------------------------
class _FastSoftmaxLastDimFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        if inp.is_cuda:
            out = _softmax_ext.softmax_lastdim_cuda(inp)
        else:
            out = torch.softmax(inp, dim=-1)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (output,) = ctx.saved_tensors
        grad_input = grad_output - (grad_output * output).sum(dim=-1, keepdim=True)
        grad_input = grad_input * output
        return grad_input


class FastSoftmaxLastDim(nn.Module):
    def forward(self, x):
        return _FastSoftmaxLastDimFunction.apply(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            FastSoftmaxLastDim(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            FastSoftmaxLastDim()
        )

    def forward(self, x):
        return self.double_conv(x)

# ---------------------------------------------------------------------
# Model definition (mirrors original I/O)
# ---------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, features):
        super().__init__()
        self.encoder1 = DoubleConv(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = DoubleConv(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = DoubleConv(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = DoubleConv(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = DoubleConv(features * 8, features * 16)

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(features * 16, features * 8)
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(features * 8, features * 4)
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(features * 4, features * 2)
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(features * 2, features)

        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.final_conv(dec1)
