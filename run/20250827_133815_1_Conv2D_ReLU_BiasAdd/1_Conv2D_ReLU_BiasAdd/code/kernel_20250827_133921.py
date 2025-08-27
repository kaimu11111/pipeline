import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------------
# Compile the fused ReLU + bias-add CUDA kernel
# ---------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void fused_relu_bias_add_kernel(const float* __restrict__ input,
                                           const float* __restrict__ bias,
                                           float* __restrict__ output,
                                           const int C,
                                           const int H,
                                           const int W,
                                           const int total_elems) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elems) return;

    const int hw = H * W;
    const int c  = (idx / hw) % C;

    float val = input[idx];
    val = (val > 0.f) ? val : 0.f;   // ReLU
    output[idx] = val + bias[c];     // bias add (broadcast across H,W)
}

torch::Tensor fused_relu_bias_add_cuda(torch::Tensor input,
                                       torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(),  "bias  must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32,
                "only float32 tensors are supported");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32,
                "only float32 tensors are supported");

    auto inp  = input.contiguous();
    auto b    = bias.contiguous();
    auto out  = torch::empty_like(inp);

    const int N = inp.size(0);
    const int C = inp.size(1);
    const int H = inp.size(2);
    const int W = inp.size(3);
    const int total_elems = inp.numel();

    constexpr int threads = 256;
    const int blocks      = (total_elems + threads - 1) / threads;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    fused_relu_bias_add_kernel<<<blocks, threads, 0, stream>>>(
        inp.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        C, H, W, total_elems);

    return out;
}
"""

cpp_source = "torch::Tensor fused_relu_bias_add_cuda(torch::Tensor input, torch::Tensor bias);"

fused_relu_bias = load_inline(
    name          = "fused_relu_bias",
    cpp_sources   = cpp_source,
    cuda_sources  = cuda_source,
    functions     = ["fused_relu_bias_add_cuda"],
    verbose       = False,
)

# ---------------------------------------------------------------------------
# Autograd wrapper for the fused kernel
# ---------------------------------------------------------------------------
class _FusedReLUAddBiasFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp: torch.Tensor, bias: torch.Tensor):
        ctx.save_for_backward(inp, bias)
        return fused_relu_bias.fused_relu_bias_add_cuda(inp, bias)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        inp, bias = ctx.saved_tensors

        # Gradient w.r.t. input
        grad_in = grad_out.clone()
        grad_in[inp <= 0] = 0  # derivative of ReLU

        # Gradient w.r.t. bias: sum over N,H,W
        grad_bias = grad_in.sum(dim=(0, 2, 3), keepdim=True)

        return grad_in, grad_bias

# Convenience Python wrapper
def fused_relu_bias_add(inp: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return _FusedReLUAddBiasFn.apply(inp, bias)

# ---------------------------------------------------------------------------
# Optimized model
# ---------------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Optimized model that keeps the high-performance PyTorch convolution
    but fuses ReLU and bias-addition with a custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = fused_relu_bias_add(x, self.bias)
        return x

# ---------------------------------------------------------------------------
# Helper functions (unchanged)
# ---------------------------------------------------------------------------
batch_size  = 32
in_channels = 32
out_channels = 64
height = width = 64
kernel_size = 3
bias_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device="cuda")]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]
