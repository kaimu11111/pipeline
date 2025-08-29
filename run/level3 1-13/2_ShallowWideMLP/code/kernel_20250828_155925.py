import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------------- #
#                           CUDA KERNEL DEFINITIONS                            #
# ---------------------------------------------------------------------------- #
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template<bool WITH_RELU>
__global__ void linear_kernel(const float* __restrict__ input,
                              const float* __restrict__ weight,
                              const float* __restrict__ bias,
                              float* __restrict__ output,
                              int batch_size,
                              int in_features,
                              int out_features)
{
    // grid.x :  ceil(out_features / blockDim.x)
    // grid.y :  batch_size
    const int row = blockIdx.y;                                    // sample index
    const int col = blockIdx.x * blockDim.x + threadIdx.x;         // feature index

    if (row >= batch_size || col >= out_features) return;

    float acc = 0.0f;
    const float* in_row  = input  + row * in_features;
    const float* w_row   = weight + col * in_features;             // weight is [out, in]
    for (int k = 0; k < in_features; ++k)
        acc += in_row[k] * w_row[k];

    acc += bias[col];
    if (WITH_RELU)
        acc = fmaxf(acc, 0.0f);

    output[row * out_features + col] = acc;
}

// ------------------------------- C++ WRAPPERS ------------------------------- //
torch::Tensor linear_forward_cuda(torch::Tensor input,
                                  torch::Tensor weight,
                                  torch::Tensor bias,
                                  bool with_relu)
{
    TORCH_CHECK(input.is_cuda(),  "input  must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(bias.is_cuda(),   "bias   must be CUDA");
    TORCH_CHECK(input.dtype()  == at::kFloat, "only float tensors are supported");
    TORCH_CHECK(weight.dtype() == at::kFloat, "only float tensors are supported");
    TORCH_CHECK(bias.dtype()   == at::kFloat, "only float tensors are supported");

    const int batch_size   = input.size(0);
    const int in_features  = input.size(1);
    const int out_features = weight.size(0);

    auto options = input.options();
    auto output  = torch::empty({batch_size, out_features}, options);

    const int threads = 256;
    const int blocks_x = (out_features + threads - 1) / threads;
    dim3 grid(blocks_x, batch_size);

    if (with_relu)
        linear_kernel<true ><<<grid, threads>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            in_features,
            out_features);
    else
        linear_kernel<false><<<grid, threads>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            in_features,
            out_features);

    return output;
}

torch::Tensor linear_relu_forward_cuda(torch::Tensor input,
                                       torch::Tensor weight,
                                       torch::Tensor bias)
{
    return linear_forward_cuda(input, weight, bias, true);
}
torch::Tensor linear_only_forward_cuda(torch::Tensor input,
                                       torch::Tensor weight,
                                       torch::Tensor bias)
{
    return linear_forward_cuda(input, weight, bias, false);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linear_relu_forward_cuda", &linear_relu_forward_cuda,
          "Linear + Bias + ReLU forward (CUDA)");
    m.def("linear_only_forward_cuda", &linear_only_forward_cuda,
          "Linear + Bias forward (CUDA)");
}
"""

# Compile / load extension
_fused_linear_cuda = load_inline(name="fused_linear_cuda",
                                 cpp_sources="",
                                 cuda_sources=cuda_source,
                                 functions=["linear_relu_forward_cuda",
                                            "linear_only_forward_cuda"],
                                 verbose=False)


# ---------------------------------------------------------------------------- #
#                        PYTORCH AUTOGRAD FUNCTION WRAPS                       #
# ---------------------------------------------------------------------------- #
class _LinearReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        input_c  = input.contiguous()
        weight_c = weight.contiguous()
        bias_c   = bias.contiguous()
        output = _fused_linear_cuda.linear_relu_forward_cuda(
            input_c, weight_c, bias_c)
        ctx.save_for_backward(input_c, weight_c, bias_c, output)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        input, weight, bias, output = ctx.saved_tensors
        # Gradient after ReLU
        relu_mask = (output > 0).type_as(grad_out)
        grad_relu = grad_out * relu_mask

        grad_input  = grad_relu @ weight          # (B,O) * (O,I) = (B,I)
        grad_weight = grad_relu.t() @ input       # (O,B) * (B,I) = (O,I)
        grad_bias   = grad_relu.sum(0)

        return grad_input, grad_weight, grad_bias


class _LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        input_c  = input.contiguous()
        weight_c = weight.contiguous()
        bias_c   = bias.contiguous()
        output = _fused_linear_cuda.linear_only_forward_cuda(
            input_c, weight_c, bias_c)
        ctx.save_for_backward(input_c, weight_c)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        input, weight = ctx.saved_tensors
        grad_input  = grad_out @ weight           # (B,O) * (O,I) = (B,I)
        grad_weight = grad_out.t() @ input        # (O,B) * (B,I) = (O,I)
        grad_bias   = grad_out.sum(0)
        return grad_input, grad_weight, grad_bias


# ---------------------------------------------------------------------------- #
#                         MODULES USING FUSED KERNELS                          #
# ---------------------------------------------------------------------------- #
class FusedLinearReLU(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = self.weight.size(1)
        bound  = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return _LinearReLUFunction.apply(x, self.weight, self.bias)


class FusedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = self.weight.size(1)
        bound  = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return _LinearFunction.apply(x, self.weight, self.bias)


# ---------------------------------------------------------------------------- #
#                                 NEW MODEL                                    #
# ---------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        super().__init__()
        layers = []
        curr_in = input_size
        for hidden in hidden_layer_sizes:
            layers.append(FusedLinearReLU(curr_in, hidden))
            curr_in = hidden
        layers.append(FusedLinear(curr_in, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ---------------------------------------------------------------------------- #
#                         HELPERS FOR BENCHMARK HARNESS                        #
# ---------------------------------------------------------------------------- #
# NOTE: these helpers mimic the originals so that external harnesses
#       expecting them will continue to work.

batch_size   = 64
input_size   = 8192
hidden_layer_sizes = [8192, 8192]
output_size  = 4096

def get_inputs():
    return [torch.rand(batch_size, input_size, device='cuda')]

def get_init_inputs():
    return [input_size, hidden_layer_sizes, output_size]
