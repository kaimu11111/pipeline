import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ----------------------------------------------------------------------
# CUDA kernel + C++ launcher
# ----------------------------------------------------------------------
source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

////////////////////////////////////////////////////////////////
// kernel
////////////////////////////////////////////////////////////////
__global__ void linear_relu_forward_kernel(
        const float* __restrict__ A,      // (batch, in_features)
        const float* __restrict__ W,      // (out_features, in_features)
        const float* __restrict__ B,      // (out_features)
        float* __restrict__ Y,            // (batch, out_features)
        int batch,
        int in_features,
        int out_features)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;   // batch  index
    const int col = blockIdx.x * blockDim.x + threadIdx.x;   // output index

    if (row < batch && col < out_features)
    {
        float acc = 0.f;
        const float* a_ptr = A + row * in_features;          // ptr to row in A
        const float* w_ptr = W + col * in_features;          // ptr to row in W
        for (int k = 0; k < in_features; ++k)
            acc += a_ptr[k] * w_ptr[k];

        acc += B[col];
        Y[row * out_features + col] = fmaxf(acc, 0.f);       // ReLU
    }
}

////////////////////////////////////////////////////////////////
// C++/CUDA launcher
////////////////////////////////////////////////////////////////
torch::Tensor linear_relu_forward_cuda(torch::Tensor A,
                                       torch::Tensor W,
                                       torch::Tensor B)
{
    TORCH_CHECK(A.is_cuda() && W.is_cuda() && B.is_cuda(), "All tensors must be CUDA");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(W.dtype() == torch::kFloat32, "W must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");

    const int batch        = A.size(0);
    const int in_features  = A.size(1);
    const int out_features = W.size(0);

    auto Y = torch::empty({batch, out_features}, A.options());

    dim3 block(16, 16);
    dim3 grid((out_features + block.x - 1) / block.x,
              (batch        + block.y - 1) / block.y);

    linear_relu_forward_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        W.data_ptr<float>(),
        B.data_ptr<float>(),
        Y.data_ptr<float>(),
        batch,
        in_features,
        out_features);

    return Y;
}
"""

# ----------------------------------------------------------------------
# C++ prototypes
# ----------------------------------------------------------------------
cpp_src = """
torch::Tensor linear_relu_forward_cuda(torch::Tensor A, torch::Tensor W, torch::Tensor B);
"""

# ----------------------------------------------------------------------
# Build extension
# ----------------------------------------------------------------------
linear_relu_ext = load_inline(
    name="linear_relu_ext",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["linear_relu_forward_cuda"],
    verbose=False
)

# ----------------------------------------------------------------------
# Autograd function (forward fused, backward via PyTorch)
# ----------------------------------------------------------------------
class LinearReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, weight, bias):
        inp_c    = inp.contiguous()
        weight_c = weight.contiguous()
        bias_c   = bias.contiguous()

        out = linear_relu_ext.linear_relu_forward_cuda(inp_c, weight_c, bias_c)

        ctx.save_for_backward(inp_c, weight_c, bias_c, out)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        inp, weight, bias, out = ctx.saved_tensors

        grad = grad_out.clone()
        grad[out == 0] = 0

        grad_inp    = grad.matmul(weight)          # (B, in_feat)
        grad_weight = grad.t().matmul(inp)         # (out_feat, in_feat)
        grad_bias   = grad.sum(0)                  # (out_feat)

        return grad_inp, grad_weight, grad_bias

# ----------------------------------------------------------------------
# nn.Module wrapper for fused layer
# ----------------------------------------------------------------------
class FusedLinearReLU(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=torch.sqrt(torch.tensor(5.0)))
        fan_in = self.weight.size(1)
        bound  = 1 / torch.sqrt(torch.tensor(float(fan_in)))
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return LinearReLUFunction.apply(x, self.weight, self.bias)

# ----------------------------------------------------------------------
# Optimised Model
# ----------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        # Convolutional backbone (unchanged)
        self.conv1     = nn.Conv2d(3,  96, 11, stride=4, padding=2)
        self.relu1     = nn.ReLU(inplace=True)
        self.maxpool1  = nn.MaxPool2d(3, stride=2)

        self.conv2     = nn.Conv2d(96, 256, 5, padding=2)
        self.relu2     = nn.ReLU(inplace=True)
        self.maxpool2  = nn.MaxPool2d(3, stride=2)

        self.conv3     = nn.Conv2d(256, 384, 3, padding=1)
        self.relu3     = nn.ReLU(inplace=True)

        self.conv4     = nn.Conv2d(384, 384, 3, padding=1)
        self.relu4     = nn.ReLU(inplace=True)

        self.conv5     = nn.Conv2d(384, 256, 3, padding=1)
        self.relu5     = nn.ReLU(inplace=True)
        self.maxpool3  = nn.MaxPool2d(3, stride=2)

        # Fused fully-connected layers
        self.fc1 = FusedLinearReLU(256 * 2 * 2, 4096)   # 256 channels × 2 × 2 spatial
        self.dropout1 = nn.Dropout(p=0.0)

        self.fc2 = FusedLinearReLU(4096, 4096)
        self.dropout2 = nn.Dropout(p=0.0)

        # Final classifier
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.maxpool1(self.relu1(self.conv1(x)))
        x = self.maxpool2(self.relu2(self.conv2(x)))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.maxpool3(self.relu5(self.conv5(x)))

        x = torch.flatten(x, 1)

        x = self.dropout1(self.fc1(x))
        x = self.dropout2(self.fc2(x))
        x = self.fc3(x)
        return x
