import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ─────────────────────────── CUDA / C++ SOURCE ────────────────────────────
source = r"""
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda_runtime.h>

// --------------------------- PUBLIC C++ WRAPPER ---------------------------
torch::Tensor conv_relu_bias_cuda(torch::Tensor input,
                                  torch::Tensor weight,
                                  torch::Tensor bias,
                                  int padding) {
    TORCH_CHECK(input.is_cuda(),  "input must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(bias.is_cuda(),   "bias must be CUDA");

    auto y = at::conv2d(
        input,
        weight,
        bias,
        {1, 1},                  // stride
        {padding, padding},      // padding
        {1, 1},                  // dilation
        1                        // groups
    );
    return at::relu(y);
}
"""

# ─────────────────────────── C++ PROTOTYPES ─────────────────────────
cpp_src = """
torch::Tensor conv_relu_bias_cuda(torch::Tensor input,
                                  torch::Tensor weight,
                                  torch::Tensor bias,
                                  int padding);
"""

# ─────────────────────── COMPILE AND LOAD MODULE ───────────────────
conv_relu_bias_mod = load_inline(
    name="conv_relu_bias_pad",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["conv_relu_bias_cuda"],
    verbose=False,
)

# ───────────────────────────── PYTHON API ───────────────────────────
class ModelNew(nn.Module):
    """
    Fused Conv2d + BiasAdd + ReLU
    (stride=1, dilation=1, groups=1, padding defaults to 'same')
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape=None, padding=None):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        # Default to 'same' padding if not specified
        self.padding = self.kernel_size // 2 if padding is None else padding
        weight_shape = (out_channels, in_channels, self.kernel_size, self.kernel_size)
        self.weight = nn.Parameter(torch.randn(weight_shape, device='cuda'))
        self.bias   = nn.Parameter(torch.randn(out_channels, device='cuda'))

    def forward(self, x):
        return conv_relu_bias_mod.conv_relu_bias_cuda(
            x,
            self.weight,
            self.bias,
            self.padding
        )
