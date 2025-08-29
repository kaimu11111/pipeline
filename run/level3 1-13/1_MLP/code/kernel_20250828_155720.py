import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------------
# CUDA / C++ Extension: fused bias-add + ReLU and bias-add (no activation)
# ---------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

template <bool RELU>
__global__ void add_bias_activation_kernel(
        const float* __restrict__ in,
        const float* __restrict__ bias,
        float* __restrict__ out,
        int rows,
        int cols) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;

    if (idx >= total) return;

    int col = idx % cols;
    float val = in[idx] + bias[col];
    if (RELU) {
        val = val > 0.f ? val : 0.f;
    }
    out[idx] = val;
}

torch::Tensor add_bias_relu_cuda(torch::Tensor input, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "input must be 2-D");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1-D");
    TORCH_CHECK(input.size(1) == bias.size(0), "bias size mismatch");

    auto rows = input.size(0);
    auto cols = input.size(1);
    auto output = torch::empty_like(input);

    const int threads = 256;
    const int total = rows * cols;
    const int blocks = (total + threads - 1) / threads;

    add_bias_activation_kernel<true><<<blocks, threads>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        cols);

    return output;
}

torch::Tensor add_bias_cuda(torch::Tensor input, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "input must be 2-D");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1-D");
    TORCH_CHECK(input.size(1) == bias.size(0), "bias size mismatch");

    auto rows = input.size(0);
    auto cols = input.size(1);
    auto output = torch::empty_like(input);

    const int threads = 256;
    const int total = rows * cols;
    const int blocks = (total + threads - 1) / threads;

    add_bias_activation_kernel<false><<<blocks, threads>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        cols);

    return output;
}
"""

cpp_decls = """
torch::Tensor add_bias_relu_cuda(torch::Tensor input, torch::Tensor bias);
torch::Tensor add_bias_cuda(torch::Tensor input, torch::Tensor bias);
"""

fused_bias_ops = load_inline(
    name="fused_bias_ops",
    cpp_sources=cpp_decls,
    cuda_sources=cuda_source,
    functions=["add_bias_relu_cuda", "add_bias_cuda"],
    verbose=False,
    with_cuda=True,
    extra_cflags=[],
    extra_ldflags=[]
)

# ---------------------------------------------------------------------------
# Python modules wrapping the custom kernels
# ---------------------------------------------------------------------------

class LinearReLU(nn.Module):
    """
    Fully–connected layer followed by ReLU, with the bias-add and activation
    fused into a single CUDA kernel.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in = self.weight.size(1)
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Matrix multiply
        out = torch.mm(x, self.weight.t())
        # Fused bias add + ReLU
        if self.bias is not None:
            out = fused_bias_ops.add_bias_relu_cuda(out, self.bias)
        else:
            out = torch.relu(out)
        return out


class LinearNoAct(nn.Module):
    """
    Fully–connected layer with bias-add fused into a CUDA kernel.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in = self.weight.size(1)
            bound = 1 / (fan_in ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.mm(x, self.weight.t())
        if self.bias is not None:
            out = fused_bias_ops.add_bias_cuda(out, self.bias)
        return out

# ---------------------------------------------------------------------------
# Optimized Model
# ---------------------------------------------------------------------------

class ModelNew(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        super().__init__()

        layers = []
        current_in = input_size
        # Hidden layers with fused Linear+ReLU
        for size in layer_sizes:
            layers.append(LinearReLU(current_in, size))
            current_in = size
        # Output layer (no activation)
        layers.append(LinearNoAct(current_in, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
