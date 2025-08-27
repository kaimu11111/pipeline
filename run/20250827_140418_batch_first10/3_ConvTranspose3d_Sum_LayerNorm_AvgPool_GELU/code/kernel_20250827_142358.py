import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------------
# Hand-written CUDA kernels: (1) Add scalar + LayerNorm  (2) GELU
# ------------------------------------------------------------------
cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

///////////////////////////////////////////////////////////////////
//  Kernel 1  : add_scalar + LayerNorm (channel-wise)             //
///////////////////////////////////////////////////////////////////
__global__ void add_scalar_layernorm_kernel(
        const float* __restrict__ in,
        float* __restrict__ out,
        const float scalar,
        const int N, const int C, const int D, const int H, const int W,
        const float eps)
{
    const int linear_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int spatial_size = D * H * W;
    const int groups      = N * spatial_size;

    if (linear_id >= groups) return;

    int n  = linear_id / spatial_size;
    int s  = linear_id % spatial_size;
    int d  =  s / (H * W);
    int hw =  s % (H * W);
    int h  = hw / W;
    int w  = hw % W;

    const long stride_spatial = (long)D * H * W;
    const long base           = (((((long)n) * C) * D + d) * H + h) * W + w;

    // compute mean
    float mean = 0.f;
    for (int c = 0; c < C; ++c)
        mean += in[base + c * stride_spatial] + scalar;
    mean /= C;

    // compute variance
    float var = 0.f;
    for (int c = 0; c < C; ++c) {
        float diff = (in[base + c * stride_spatial] + scalar) - mean;
        var += diff * diff;
    }
    var /= C;
    float inv_std = rsqrtf(var + eps);

    // write normalized output
    for (int c = 0; c < C; ++c) {
        float val = (in[base + c * stride_spatial] + scalar - mean) * inv_std;
        out[base + c * stride_spatial] = val;
    }
}

///////////////////////////////////////////////////////////////////
//  Kernel 2  : Fast GELU (tanh approximation)                    //
///////////////////////////////////////////////////////////////////
__device__ __forceinline__ float fast_gelu(float x) {
    const float kAlpha = 0.7978845608028654f;       // sqrt(2/pi)
    return 0.5f * x * (1.f + tanhf(kAlpha * (x + 0.044715f * x * x * x)));
}

__global__ void gelu_kernel(const float* in, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    out[idx] = fast_gelu(in[idx]);
}

///////////////////////////////////////////////////////////////////
//  Host wrappers                                                 //
///////////////////////////////////////////////////////////////////
torch::Tensor add_scalar_layernorm_cuda(torch::Tensor input,
                                        float scalar,
                                        float eps)
{
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat,
                "Only float tensors are supported");

    auto output = torch::empty_like(input);

    const int N = input.size(0);
    const int C = input.size(1);
    const int D = input.size(2);
    const int H = input.size(3);
    const int W = input.size(4);

    const int groups = N * D * H * W;
    const int threads = 256;
    const int blocks  = (groups + threads - 1) / threads;

    add_scalar_layernorm_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        scalar, N, C, D, H, W, eps);

    return output;
}

torch::Tensor gelu_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat,
                "Only float tensors are supported");

    auto output = torch::empty_like(input);

    const int size    = input.numel();
    const int threads = 256;
    const int blocks  = (size + threads - 1) / threads;

    gelu_kernel<<<blocks, threads>>>(input.data_ptr<float>(),
                                     output.data_ptr<float>(),
                                     size);
    return output;
}

///////////////////////////////////////////////////////////////////
//  PyBind                                                        //
///////////////////////////////////////////////////////////////////
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_scalar_layernorm_cuda", &add_scalar_layernorm_cuda,
          "Add scalar + LayerNorm along channel dimension (CUDA)");
    m.def("gelu_cuda", &gelu_cuda,
          "Fast GELU activation (CUDA)");
}
"""

cpp_decl = """
torch::Tensor add_scalar_layernorm_cuda(torch::Tensor input, float scalar, float eps);
torch::Tensor gelu_cuda(torch::Tensor input);
"""

# Build / load kernels
_ops = load_inline(
    name="custom_3d_ops",
    cpp_sources=cpp_decl,
    cuda_sources=cuda_src,
    functions=["add_scalar_layernorm_cuda", "gelu_cuda"],
    verbose=False,
)

# ------------------------------------------------------------------
#                   Optimised PyTorch Model (ModelNew)
# ------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Optimised model with fused CUDA kernels:
      1) conv_transpose3d (PyTorch)
      2) fused add-scalar + LayerNorm  (custom CUDA)
      3) avg_pool3d                    (PyTorch)
      4) custom GELU                   (custom CUDA)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, output_padding, sum_weight, norm_shape,
                 pool_kernel_size, eps=1e-5):
        super().__init__()

        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )

        # keep scalar as parameter (not used in autograd in fused op)
        self.sum_weight = nn.Parameter(torch.tensor(float(sum_weight),
                                                    dtype=torch.float32))

        # affine parameters for LayerNorm
        self.ln_weight = nn.Parameter(torch.ones(norm_shape, dtype=torch.float32))
        self.ln_bias   = nn.Parameter(torch.zeros(norm_shape, dtype=torch.float32))

        self.avg_pool = nn.AvgPool3d(kernel_size=pool_kernel_size)

        self.eps = float(eps)  # layer-norm epsilon

    def forward(self, x):
        # (1) conv transpose
        x = self.conv_transpose(x)

        # (2) add scalar + layernorm (channels)
        x = _ops.add_scalar_layernorm_cuda(x, float(self.sum_weight.detach()), self.eps)

        # apply affine transform of LayerNorm
        x = x * self.ln_weight.view(1, -1, 1, 1, 1) + self.ln_bias.view(1, -1, 1, 1, 1)

        # (3) average pooling
        x = self.avg_pool(x)

        # (4) GELU
        x = _ops.gelu_cuda(x)
        return x

# ------------------- helper functions for testing -----------------
def get_inputs():
    batch_size = 16
    in_channels = 16
    depth, height, width = 8, 16, 16
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]

def get_init_inputs():
    batch_size = 16
    in_channels = 16
    out_channels = 32
    kernel_size = (3, 3, 3)
    stride = (2, 2, 2)
    padding = (1, 1, 1)
    output_padding = (1, 1, 1)
    sum_weight = 1.0
    norm_shape = (out_channels,)
    pool_kernel_size = (2, 2, 2)
    return [in_channels, out_channels, kernel_size, stride, padding,
            output_padding, sum_weight, norm_shape, pool_kernel_size]
