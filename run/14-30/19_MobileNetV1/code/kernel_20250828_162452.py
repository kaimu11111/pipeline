import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------
# 1. CUDA kernels + pybind11 wrapper
# ---------------------------------------------------------------------
source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void relu_forward_kernel(const scalar_t* __restrict__ inp,
                                    scalar_t* __restrict__ out,
                                    size_t numel) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        const scalar_t x = inp[idx];
        out[idx] = x > scalar_t(0) ? x : scalar_t(0);
    }
}

template <typename scalar_t>
__global__ void relu_backward_kernel(const scalar_t* __restrict__ grad_out,
                                     const scalar_t* __restrict__ inp,
                                     scalar_t* __restrict__ grad_in,
                                     size_t numel) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        const scalar_t x = inp[idx];
        grad_in[idx] = x > scalar_t(0) ? grad_out[idx] : scalar_t(0);
    }
}

torch::Tensor relu_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const size_t numel = input.numel();
    const int threads = 256;
    const int blocks  = static_cast<int>((numel + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "relu_forward_cuda", [&] {
        relu_forward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            numel);
    });
    return output;
}

torch::Tensor relu_backward(torch::Tensor grad_out, torch::Tensor input) {
    auto grad_in = torch::empty_like(input);
    const size_t numel = input.numel();
    const int threads = 256;
    const int blocks  = static_cast<int>((numel + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "relu_backward_cuda", [&] {
        relu_backward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            grad_out.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            grad_in.data_ptr<scalar_t>(),
            numel);
    });
    return grad_in;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("relu_forward",  &relu_forward,  "Fast ReLU forward (CUDA)");
    m.def("relu_backward", &relu_backward, "Fast ReLU backward (CUDA)");
}
"""

# ---------------------------------------------------------------------
# 2. C++ prototypes (visible to the loader but not used for compilation)
# ---------------------------------------------------------------------
cpp_src = r"""
torch::Tensor relu_forward(torch::Tensor input);
torch::Tensor relu_backward(torch::Tensor grad_out, torch::Tensor input);
"""

# ---------------------------------------------------------------------
# 3. Build & load the extension
# ---------------------------------------------------------------------
relu_ops = load_inline(
    name="fast_relu",
    cpp_sources=cpp_src,
    cuda_sources=source,
    with_cuda=True,
    verbose=False,
)

# ---------------------------------------------------------------------
# 4. PyTorch autograd binding
# ---------------------------------------------------------------------
class _FastReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return relu_ops.relu_forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return relu_ops.relu_backward(grad_output.contiguous(), x)

class CUDAReLU(nn.Module):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace  # kept for API compatibility

    def forward(self, x):
        # In-place variant is not supported by the custom kernel;
        # we still expose the flag to stay drop-in compatible.
        return _FastReLUFunction.apply(x)

# ---------------------------------------------------------------------
# 5. MobileNet-V1 using the custom CUDA ReLU
# ---------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, num_classes: int = 1000, input_channels: int = 3, alpha: float = 1.0):
        super().__init__()
        relu = CUDAReLU  # alias for brevity

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                relu(inplace=False)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                relu(inplace=False),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                relu(inplace=False),
            )

        self.features = nn.Sequential(
            conv_bn(input_channels, int(32 * alpha), 2),
            conv_dw(int(32 * alpha), int(64 * alpha), 1),
            conv_dw(int(64 * alpha), int(128 * alpha), 2),
            conv_dw(int(128 * alpha), int(128 * alpha), 1),
            conv_dw(int(128 * alpha), int(256 * alpha), 2),
            conv_dw(int(256 * alpha), int(256 * alpha), 1),
            conv_dw(int(256 * alpha), int(512 * alpha), 2),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(1024 * alpha), 2),
            conv_dw(int(1024 * alpha), int(1024 * alpha), 1),
            nn.AvgPool2d(7),
        )

        self.classifier = nn.Linear(int(1024 * alpha), num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
