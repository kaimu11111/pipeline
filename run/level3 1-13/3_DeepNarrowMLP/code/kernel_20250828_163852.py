import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------
# CUDA/C++ source for fused Linear(+Bias)[+ReLU] operator
# ---------------------------------------------------------------------
cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template<bool RELU>
__global__ void linear_bias_activation_kernel(const float* __restrict__ input,
                                              const float* __restrict__ weight,
                                              const float* __restrict__ bias,
                                              float* __restrict__ output,
                                              int batch,
                                              int in_features,
                                              int out_features) {
    int out_idx   = blockIdx.x * blockDim.x + threadIdx.x;   // column  (output feature)
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;   // row     (batch sample)

    if (out_idx >= out_features || batch_idx >= batch) return;

    const float* w_row  = weight + out_idx * in_features;        // W[out_idx, :]
    const float* x_row  = input  + batch_idx * in_features;      // X[batch_idx, :]

    float acc = 0.f;
    for (int k = 0; k < in_features; ++k) {
        acc += w_row[k] * x_row[k];
    }

    acc += bias[out_idx];
    if (RELU && acc < 0.f) acc = 0.f;

    output[batch_idx * out_features + out_idx] = acc;
}

// ------------------------- WRAPPERS -------------------------------
torch::Tensor linear_bias_relu_forward_cuda(torch::Tensor input,
                                            torch::Tensor weight,
                                            torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda() && weight.is_cuda() && bias.is_cuda(),
                "All tensors must be on CUDA devices");
    TORCH_CHECK(input.scalar_type()  == torch::kFloat32 &&
                weight.scalar_type() == torch::kFloat32 &&
                bias.scalar_type()   == torch::kFloat32,
                "Only float32 tensors are supported");

    const int batch       = input.size(0);
    const int in_features = input.size(1);
    const int out_features= weight.size(0);

    auto output = torch::empty({batch, out_features}, input.options());

    dim3 block(16, 16);
    dim3 grid((out_features + block.x - 1) / block.x,
              (batch       + block.y - 1) / block.y);

    linear_bias_activation_kernel<true><<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        in_features,
        out_features);

    return output;
}

torch::Tensor linear_bias_forward_cuda(torch::Tensor input,
                                       torch::Tensor weight,
                                       torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda() && weight.is_cuda() && bias.is_cuda(),
                "All tensors must be on CUDA devices");
    TORCH_CHECK(input.scalar_type()  == torch::kFloat32 &&
                weight.scalar_type() == torch::kFloat32 &&
                bias.scalar_type()   == torch::kFloat32,
                "Only float32 tensors are supported");

    const int batch       = input.size(0);
    const int in_features = input.size(1);
    const int out_features= weight.size(0);

    auto output = torch::empty({batch, out_features}, input.options());

    dim3 block(16, 16);
    dim3 grid((out_features + block.x - 1) / block.x,
              (batch       + block.y - 1) / block.y);

    linear_bias_activation_kernel<false><<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        in_features,
        out_features);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linear_bias_relu_forward", &linear_bias_relu_forward_cuda,
          "Linear + Bias + ReLU forward (CUDA)");
    m.def("linear_bias_forward",      &linear_bias_forward_cuda,
          "Linear + Bias forward (CUDA)");
}
"""

cpp_decls = """
torch::Tensor linear_bias_relu_forward_cuda(torch::Tensor input,
                                            torch::Tensor weight,
                                            torch::Tensor bias);
torch::Tensor linear_bias_forward_cuda(torch::Tensor input,
                                       torch::Tensor weight,
                                       torch::Tensor bias);
"""

fused_linear_cuda = load_inline(
    name="fused_linear_cuda",
    cpp_sources=cpp_decls,
    cuda_sources=cuda_src,
    functions=[
        "linear_bias_relu_forward",
        "linear_bias_forward",
    ],
    verbose=False,
)

# ---------------------------------------------------------------------
# Python wrappers for the fused kernels
# ---------------------------------------------------------------------
class _FusedLinearBase(nn.Module):
    def __init__(self, in_features: int, out_features: int, apply_relu: bool):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.apply_relu   = apply_relu

        # Parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device='cuda'))
        self.bias   = nn.Parameter(torch.empty(out_features, device='cuda'))

        # Kaiming uniform initialization (same as nn.Linear)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.apply_relu:
            return fused_linear_cuda.linear_bias_relu_forward(x.contiguous(), self.weight, self.bias)
        else:
            return fused_linear_cuda.linear_bias_forward(x.contiguous(), self.weight, self.bias)


class FusedLinearReLU(_FusedLinearBase):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, apply_relu=True)


class FusedLinear(_FusedLinearBase):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, apply_relu=False)

# ---------------------------------------------------------------------
# Optimised model definition
# ---------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        super().__init__()
        layers = []
        current_size = input_size
        for hidden_size in hidden_layer_sizes:
            layers.append(FusedLinearReLU(current_size, hidden_size))
            current_size = hidden_size
        layers.append(FusedLinear(current_size, output_size))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

# ---------------------------------------------------------------------
# Helper functions matching the original interface
# ---------------------------------------------------------------------
batch_size = 512
input_size = 4096
hidden_layer_sizes = [256] * 16
output_size = 4096

def get_inputs():
    return [torch.rand(batch_size, input_size, device='cuda', dtype=torch.float32)]

def get_init_inputs():
    return [input_size, hidden_layer_sizes, output_size]
