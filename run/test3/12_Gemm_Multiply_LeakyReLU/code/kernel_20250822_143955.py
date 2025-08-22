import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# --------------------------------------------------------------------
# CUDA kernels: bias + LeakyReLU + scalar multiply
# --------------------------------------------------------------------
source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

////////////////////////////////////////////////////////////////////////////////
// Element-wise kernel
////////////////////////////////////////////////////////////////////////////////
__global__ void bias_act_mul_kernel(
    const float* __restrict__ X,     // [M*N] – pre-computed matmul result
    const float* __restrict__ bias,  // [N]
    float* __restrict__ Y,           // [M*N] – output
    int M, int N,
    float multiplier,
    float negative_slope)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = M * N;

    for (int idx = tid; idx < total; idx += gridDim.x * blockDim.x)
    {
        float val = X[idx] + bias[idx % N];                 // add bias
        val = (val >= 0.0f) ? val : val * negative_slope;   // LeakyReLU
        val *= multiplier;                                  // scale
        Y[idx] = val;
    }
}

////////////////////////////////////////////////////////////////////////////////
// C++ interface
////////////////////////////////////////////////////////////////////////////////
torch::Tensor bias_act_mul_cuda(
    torch::Tensor X,
    torch::Tensor bias,
    float multiplier,
    float negative_slope)
{
    TORCH_CHECK(X.is_cuda() && bias.is_cuda(), "Tensors must be on CUDA");
    TORCH_CHECK(X.dtype() == torch::kFloat32 && bias.dtype() == torch::kFloat32,
                "Only float32 supported");

    const int64_t M = X.size(0);
    const int64_t N = X.size(1);
    TORCH_CHECK(bias.numel() == N, "Bias shape mismatch");

    auto Y = torch::empty_like(X);

    const int threads = 256;
    const int blocks  = (M * N + threads - 1) / threads;

    bias_act_mul_kernel<<<blocks, threads>>>(
        X.data_ptr<float>(),
        bias.data_ptr<float>(),
        Y.data_ptr<float>(),
        static_cast<int>(M),
        static_cast<int>(N),
        multiplier,
        negative_slope);

    return Y;
}
"""

# --------------------------------------------------------------------
# C++ prototypes
# --------------------------------------------------------------------
cpp_src = """
torch::Tensor bias_act_mul_cuda(
    torch::Tensor X,
    torch::Tensor bias,
    float multiplier,
    float negative_slope);
"""

# --------------------------------------------------------------------
# Build & load
# --------------------------------------------------------------------
bias_act_ext = load_inline(
    name="bias_act_mul",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["bias_act_mul_cuda"],
    verbose=False,
)

# --------------------------------------------------------------------
# PyTorch module
# --------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Performs:   y = (x @ W.T + b) → LeakyReLU → scale (multiplier)
    GEMM is done by cuBLAS (torch.mm); everything else by our CUDA kernel.
    """
    def __init__(self, in_features, out_features,
                 multiplier=1.0, negative_slope=0.01):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.empty(out_features))
        self.multiplier      = float(multiplier)
        self.negative_slope  = float(negative_slope)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = in_features
        bound = 1.0 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if not x.is_cuda:
            x = x.cuda()

        # GEMM via cuBLAS
        mat = torch.mm(x, self.weight.t())

        # Fused bias + activation + scaling via custom kernel
        y = bias_act_ext.bias_act_mul_cuda(
            mat,
            self.bias.cuda(),
            self.multiplier,
            self.negative_slope
        )
        return y
