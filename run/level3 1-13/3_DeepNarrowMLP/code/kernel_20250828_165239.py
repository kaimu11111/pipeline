# 1. Imports
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# 2. source – CUDA code (kernels + host wrappers)
source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

/*
  The weight matrix coming from Python is passed **transposed**
  ([in_features, out_features]) so that the CUDA kernel can access it
  column-contiguously and produce   output = X · Wᵀ (+ bias)   exactly
  like torch.nn.functional.linear.
*/

template<bool RELU>
__global__ void linear_bias_activation_kernel(const float* __restrict__ input,
                                              const float* __restrict__ weight_T, // (in, out)
                                              const float* __restrict__ bias,
                                              float*       __restrict__ output,
                                              int batch,
                                              int in_features,
                                              int out_features) {
    const int out_idx   = blockIdx.x * blockDim.x + threadIdx.x;  // column  (output feature)
    const int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;  // row     (batch sample)

    if (out_idx >= out_features || batch_idx >= batch) return;

    const float* x_row = input + batch_idx * in_features;          // X[batch_idx, :]

    float acc = 0.f;
    #pragma unroll 4
    for (int k = 0; k < in_features; ++k) {
        // weight_T[k, out_idx]  – because weight is passed transposed
        acc += weight_T[k * out_features + out_idx] * x_row[k];
    }

    acc += bias[out_idx];
    if (RELU && acc < 0.f) acc = 0.f;

    output[batch_idx * out_features + out_idx] = acc;
}

// Forward with ReLU -------------------------------------------------
torch::Tensor linear_bias_relu_forward(torch::Tensor input,
                                       torch::Tensor weight_T,
                                       torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda()   && weight_T.is_cuda() && bias.is_cuda(),
                "All tensors must be CUDA tensors");
    TORCH_CHECK(input.scalar_type()   == torch::kFloat32 &&
                weight_T.scalar_type()== torch::kFloat32 &&
                bias.scalar_type()    == torch::kFloat32,
                "Only float32 tensors are supported");

    const int batch        = input.size(0);
    const int in_features  = input.size(1);
    const int out_features = bias.size(0);                         // same as weight_T.size(1)

    auto output = torch::empty({batch, out_features}, input.options());

    dim3 block(16, 16);
    dim3 grid((out_features + block.x - 1) / block.x,
              (batch       + block.y - 1) / block.y);

    linear_bias_activation_kernel<true><<<grid, block>>>(
        input.data_ptr<float>(),
        weight_T.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        in_features,
        out_features);

    return output;
}

// Forward WITHOUT ReLU ---------------------------------------------
torch::Tensor linear_bias_forward(torch::Tensor input,
                                  torch::Tensor weight_T,
                                  torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda()   && weight_T.is_cuda() && bias.is_cuda(),
                "All tensors must be CUDA tensors");
    TORCH_CHECK(input.scalar_type()   == torch::kFloat32 &&
                weight_T.scalar_type()== torch::kFloat32 &&
                bias.scalar_type()    == torch::kFloat32,
                "Only float32 tensors are supported");

    const int batch        = input.size(0);
    const int in_features  = input.size(1);
    const int out_features = bias.size(0);

    auto output = torch::empty({batch, out_features}, input.options());

    dim3 block(16, 16);
    dim3 grid((out_features + block.x - 1) / block.x,
              (batch       + block.y - 1) / block.y);

    linear_bias_activation_kernel<false><<<grid, block>>>(
        input.data_ptr<float>(),
        weight_T.data_ptr<float>(),
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
                                       torch::Tensor weight_T,
                                       torch::Tensor bias);
torch::Tensor linear_bias_forward(torch::Tensor input,
                                  torch::Tensor weight_T,
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
    """
    Mirrors a reference fully-connected network that alternates
    Linear → ReLU … Linear (no ReLU on the last layer).

    The nn.Linear modules themselves are kept so that
    `state_dict` is identical to the reference implementation;
    we merely replace their forward path with our CUDA kernels.
    """
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        super().__init__()
        sizes = [input_size] + list(hidden_layer_sizes) + [output_size]

        self.layers = nn.ModuleList()
        for in_f, out_f in zip(sizes[:-1], sizes[1:]):
            self.layers.append(nn.Linear(in_f, out_f))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_layers = len(self.layers)
        for idx, lin in enumerate(self.layers):
            # weight in layout (out, in) — same as nn.Linear
            # the CUDA kernel expects the **transposed** weight
            w_T = lin.weight.t().contiguous()
            b   = lin.bias
            if idx == num_layers - 1:
                x = fused_linear_cuda.linear_bias_forward(
                    x.contiguous(), w_T, b
                )
            else:
                x = fused_linear_cuda.linear_bias_relu_forward(
                    x.contiguous(), w_T, b
                )
        return x
