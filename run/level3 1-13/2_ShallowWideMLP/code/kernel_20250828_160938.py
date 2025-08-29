# 1. Imports ───────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Provide a default so external helpers never raise NameError
hidden_layer_sizes = [128, 64]


# 2. source – CUDA kernel + C++ wrapper ────────────────────────────────
source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------
//                               KERNEL
// ---------------------------------------------------------------------
template<bool WITH_RELU>
__global__ void linear_kernel(const float* __restrict__ input,
                              const float* __restrict__ weight,
                              const float* __restrict__ bias,
                              float* __restrict__ output,
                              int batch_size,
                              int in_features,
                              int out_features)
{
    const int row = blockIdx.y;                                    // batch index
    const int col = blockIdx.x * blockDim.x + threadIdx.x;         // feature index
    if (row >= batch_size || col >= out_features) return;

    float acc = 0.f;
    const float* in_row = input  + row * in_features;
    const float* w_row  = weight + col * in_features;              // weight is [out, in]
    #pragma unroll 4
    for (int k = 0; k < in_features; ++k)
        acc += in_row[k] * w_row[k];

    acc += bias[col];
    if (WITH_RELU)  acc = fmaxf(acc, 0.0f);
    output[row * out_features + col] = acc;
}

// ---------------------------------------------------------------------
//                         HOST ‑ SIDE WRAPPERS
// ---------------------------------------------------------------------
static torch::Tensor linear_forward_cuda(torch::Tensor input,
                                         torch::Tensor weight,
                                         torch::Tensor bias,
                                         bool with_relu)
{
    TORCH_CHECK(input.is_cuda()  && weight.is_cuda() && bias.is_cuda(),
                "All tensors must reside on CUDA");
    TORCH_CHECK(input.dtype() == at::kFloat &&
                weight.dtype() == at::kFloat &&
                bias.dtype()   == at::kFloat,
                "Only float32 tensors are supported");

    const int B = input.size(0);
    const int I = input.size(1);
    const int O = weight.size(0);

    auto output = torch::empty({B, O}, input.options());

    const int threads  = 256;
    const int blocks_x = (O + threads - 1) / threads;
    dim3 grid(blocks_x, B);

    if (with_relu)
        linear_kernel<true><<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.data_ptr<float>(),
            output.data_ptr<float>(),
            B, I, O);
    else
        linear_kernel<false><<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.data_ptr<float>(),
            output.data_ptr<float>(),
            B, I, O);

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
"""


# 3. cpp_src – forward declarations for exposed symbols ───────────────
cpp_src = r"""
#include <torch/extension.h>

torch::Tensor linear_relu_forward_cuda(torch::Tensor input,
                                       torch::Tensor weight,
                                       torch::Tensor bias);

torch::Tensor linear_only_forward_cuda(torch::Tensor input,
                                       torch::Tensor weight,
                                       torch::Tensor bias);
"""


# 4. load_inline call ─────────────────────────────────────────────────
_fused_linear_cuda = load_inline(
    name="fused_linear_cuda",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["linear_relu_forward_cuda",
               "linear_only_forward_cuda"],
    verbose=False
)


# 5. class ModelNew ───────────────────────────────────────────────────
class ModelNew(nn.Module):
    # --------------------------------------------------------------
    # Nested autograd Functions
    # --------------------------------------------------------------
    class _LinearReLUFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, weight, bias):
            input_c, weight_c, bias_c = input.contiguous(), weight.contiguous(), bias.contiguous()
            out = _fused_linear_cuda.linear_relu_forward_cuda(input_c, weight_c, bias_c)
            ctx.save_for_backward(input_c, weight_c, bias_c, out)
            return out

        @staticmethod
        def backward(ctx, grad_out):
            inp, w, b, out = ctx.saved_tensors
            grad_relu   = grad_out * (out > 0).type_as(grad_out)
            grad_input  = grad_relu @ w
            grad_weight = grad_relu.t() @ inp
            grad_bias   = grad_relu.sum(0)
            return grad_input, grad_weight, grad_bias

    class _LinearFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, weight, bias):
            input_c, weight_c, bias_c = input.contiguous(), weight.contiguous(), bias.contiguous()
            out = _fused_linear_cuda.linear_only_forward_cuda(input_c, weight_c, bias_c)
            ctx.save_for_backward(input_c, weight_c)
            return out

        @staticmethod
        def backward(ctx, grad_out):
            inp, w = ctx.saved_tensors
            grad_input  = grad_out @ w
            grad_weight = grad_out.t() @ inp
            grad_bias   = grad_out.sum(0)
            return grad_input, grad_weight, grad_bias

    # --------------------------------------------------------------
    # Simple Modules built on top of the autograd Functions
    # --------------------------------------------------------------
    class FusedLinearReLU(nn.Module):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
            self.bias   = nn.Parameter(torch.empty(out_features))
            self.reset_parameters()

        def reset_parameters(self):
            nn.init.kaiming_uniform_(self.weight, a=(5 ** 0.5))
            bound = 1 / (self.weight.size(1) ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

        def forward(self, x):
            return ModelNew._LinearReLUFunction.apply(x, self.weight, self.bias)

    class FusedLinear(nn.Module):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
            self.bias   = nn.Parameter(torch.empty(out_features))
            self.reset_parameters()

        def reset_parameters(self):
            nn.init.kaiming_uniform_(self.weight, a=(5 ** 0.5))
            bound = 1 / (self.weight.size(1) ** 0.5)
            nn.init.uniform_(self.bias, -bound, bound)

        def forward(self, x):
            return ModelNew._LinearFunction.apply(x, self.weight, self.bias)

    # --------------------------------------------------------------
    # ModelNew initialisation / forward
    # --------------------------------------------------------------
    def __init__(self, input_size, hidden_layer_sizes=None, output_size=1):
        super().__init__()
        # Ensure we always have an iterable list of hidden sizes
        hidden_layer_sizes = list(hidden_layer_sizes) if hidden_layer_sizes is not None else []
        layers = []
        in_f = input_size
        for h in hidden_layer_sizes:
            layers.append(self.FusedLinearReLU(in_f, h))
            in_f = h
        layers.append(self.FusedLinear(in_f, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
