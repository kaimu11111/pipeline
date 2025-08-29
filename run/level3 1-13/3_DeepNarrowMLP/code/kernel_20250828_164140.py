# 1. Imports
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# 2. source – CUDA code (kernels + host wrappers)
source = r"""
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

// Forward with ReLU
torch::Tensor linear_bias_relu_forward(torch::Tensor input,
                                       torch::Tensor weight,
                                       torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda() && weight.is_cuda() && bias.is_cuda(),
                "All tensors must be on CUDA devices");
    TORCH_CHECK(input.scalar_type()  == torch::kFloat32 &&
                weight.scalar_type() == torch::kFloat32 &&
                bias.scalar_type()   == torch::kFloat32,
                "Only float32 tensors are supported");

    const int batch        = input.size(0);
    const int in_features  = input.size(1);
    const int out_features = weight.size(0);

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

// Forward without ReLU
torch::Tensor linear_bias_forward(torch::Tensor input,
                                  torch::Tensor weight,
                                  torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda() && weight.is_cuda() && bias.is_cuda(),
                "All tensors must be on CUDA devices");
    TORCH_CHECK(input.scalar_type()  == torch::kFloat32 &&
                weight.scalar_type() == torch::kFloat32 &&
                bias.scalar_type()   == torch::kFloat32,
                "Only float32 tensors are supported");

    const int batch        = input.size(0);
    const int in_features  = input.size(1);
    const int out_features = weight.size(0);

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
"""

# 3. cpp_src – prototypes for all exposed kernels
cpp_src = """
torch::Tensor linear_bias_relu_forward(torch::Tensor input,
                                       torch::Tensor weight,
                                       torch::Tensor bias);
torch::Tensor linear_bias_forward(torch::Tensor input,
                                  torch::Tensor weight,
                                  torch::Tensor bias);
"""

# 4. Single load_inline call
fused_linear_cuda = load_inline(
    name="fused_linear_cuda",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["linear_bias_relu_forward", "linear_bias_forward"],
    verbose=False,
)

# 5. class ModelNew – uses fused kernels
class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        super().__init__()
        layer_sizes = [input_size] + list(hidden_layer_sizes) + [output_size]

        self.weights = nn.ParameterList()
        self.biases  = nn.ParameterList()

        for in_f, out_f in zip(layer_sizes[:-1], layer_sizes[1:]):
            w = nn.Parameter(torch.empty(out_f, in_f, device='cuda'))
            b = nn.Parameter(torch.empty(out_f, device='cuda'))

            # Kaiming uniform initialisation (as in nn.Linear)
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(b, -bound, bound)

            self.weights.append(w)
            self.biases.append(b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_layers = len(self.weights)
        for idx, (w, b) in enumerate(zip(self.weights, self.biases)):
            if idx == num_layers - 1:
                x = fused_linear_cuda.linear_bias_forward(x.contiguous(), w, b)
            else:
                x = fused_linear_cuda.linear_bias_relu_forward(x.contiguous(), w, b)
        return x
