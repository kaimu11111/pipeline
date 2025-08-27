import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -------------------------------------------------------------------------
# Fused ReLU + BiasAdd CUDA kernel
# -------------------------------------------------------------------------
cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_bias_add_kernel(const float* __restrict__ in,
                                     const float* __restrict__ bias,
                                     float* __restrict__ out,
                                     const int channels,
                                     const int hw,
                                     const int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int c = (idx / hw) % channels;        // derive channel index
    float v = in[idx];
    v = v > 0.f ? v : 0.f;                // ReLU
    out[idx] = v + bias[c];               // add bias
}

torch::Tensor relu_bias_add_cuda(torch::Tensor input,
                                 torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(),  "bias  must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "only float tensors are supported");
    TORCH_CHECK(bias.scalar_type()  == at::kFloat, "only float tensors are supported");

    input = input.contiguous();
    bias  = bias.contiguous();

    const int C      = input.size(1);
    const int H      = input.size(2);
    const int W      = input.size(3);
    const int hw     = H * W;
    const int total  = input.numel();

    auto output = torch::empty_like(input);

    const int threads = 256;
    const int blocks  = (total + threads - 1) / threads;

    relu_bias_add_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        C,
        hw,
        total
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("relu_bias_add_cuda", &relu_bias_add_cuda,
          "Fused ReLU + BiasAdd (CUDA)");
}
"""

# Load / compile the CUDA extension
relu_bias_add = load_inline(
    name="relu_bias_add",
    cpp_sources="",
    cuda_sources=cuda_src,
    functions=["relu_bias_add_cuda"],
    with_cuda=True,
    verbose=False,
)

# -------------------------------------------------------------------------
# Optimised Model definition
# -------------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Optimised model that performs Conv2d followed by a fused ReLU + BiasAdd.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))           # (C,1,1)
        self.fused_op = relu_bias_add

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_op.relu_bias_add_cuda(x, self.bias)
        return x

# -------------------------------------------------------------------------
# Helper functions (same signatures as original script)
# -------------------------------------------------------------------------
batch_size   = 32
in_channels  = 32
out_channels = 64
height = width = 64
kernel_size  = 3
bias_shape   = (out_channels, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device="cuda")]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]
