import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------------
# CUDA kernels + host wrappers
# ------------------------------------------------------------------
source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

///////////////////////////////////////////////////////////////////
// fast GELU (tanh approximation)
///////////////////////////////////////////////////////////////////
__device__ __forceinline__ float fast_gelu(float x) {
    const float kAlpha = 0.7978845608028654f;       // sqrt(2/pi)
    return 0.5f * x * (1.f + tanhf(kAlpha * (x + 0.044715f * x * x * x)));
}

///////////////////////////////////////////////////////////////////
// Kernel: add_scalar + LayerNorm over the LAST dimension
///////////////////////////////////////////////////////////////////
__global__ void add_scalar_layernorm_kernel_lastdim(
        const float* __restrict__ in,
        float* __restrict__ out,
        const float  scalar,
        const int    features,      // size of last dim
        const int    groups,        // prod of all leading dims
        const float  eps)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= groups) return;

    const int offset = gid * features;

    // compute mean
    float mean = 0.f;
    for (int i = 0; i < features; ++i)
        mean += in[offset + i] + scalar;
    mean /= features;

    // compute variance
    float var = 0.f;
    for (int i = 0; i < features; ++i) {
        float diff = (in[offset + i] + scalar) - mean;
        var += diff * diff;
    }
    var /= features;
    float inv_std = rsqrtf(var + eps);

    // write normalized output
    for (int i = 0; i < features; ++i) {
        float val = (in[offset + i] + scalar - mean) * inv_std;
        out[offset + i] = val;
    }
}

///////////////////////////////////////////////////////////////////
// Kernel: element-wise fast GELU
///////////////////////////////////////////////////////////////////
__global__ void gelu_kernel(const float* in, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    out[idx] = fast_gelu(in[idx]);
}

///////////////////////////////////////////////////////////////////
// Host wrappers
///////////////////////////////////////////////////////////////////
torch::Tensor add_scalar_layernorm_cuda(torch::Tensor input,
                                        float scalar,
                                        float eps)
{
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(input.scalar_type() == torch::kFloat,
                "only float32 tensors supported");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    const int64_t features = input.size(-1);            // width dimension
    const int64_t groups   = input.numel() / features;  // N*C*D*H

    auto output = torch::empty_like(input);

    const int threads = 256;
    const int blocks  = (groups + threads - 1) / threads;

    add_scalar_layernorm_kernel_lastdim<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        scalar,
        (int)features,
        (int)groups,
        eps
    );
    return output;
}

torch::Tensor gelu_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(input.scalar_type() == torch::kFloat,
                "only float32 tensors supported");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    auto output = torch::empty_like(input);

    const int64_t size = input.numel();
    const int threads  = 256;
    const int blocks   = (size + threads - 1) / threads;

    gelu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        (int)size
    );
    return output;
}
"""

# ------------------------------------------------------------------
# Function prototypes for bindings
# ------------------------------------------------------------------
cpp_src = """
torch::Tensor add_scalar_layernorm_cuda(torch::Tensor input, float scalar, float eps);
torch::Tensor gelu_cuda(torch::Tensor input);
"""

# ------------------------------------------------------------------
# Build / load kernels (one extension)
# ------------------------------------------------------------------
_ops = load_inline(
    name="custom_3d_ops_fixed",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["add_scalar_layernorm_cuda", "gelu_cuda"],
    verbose=False,
)

# ------------------------------------------------------------------
#                   Optimised PyTorch Model
# ------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Sequence:
      1) ConvTranspose3d  (PyTorch)
      2) add-scalar + LayerNorm over spatial width axis (CUDA)
      3) AvgPool3d        (PyTorch)
      4) fast GELU        (CUDA)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, output_padding, sum_weight, norm_shape,
                 pool_kernel_size, eps=1e-5):
        super().__init__()

        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )

        # scalar added before normalisation
        self.sum_weight = nn.Parameter(torch.tensor(float(sum_weight),
                                                    dtype=torch.float32))

        # affine parameters for LayerNorm (matches width dimension)
        self.ln_weight = nn.Parameter(torch.ones(norm_shape, dtype=torch.float32))
        self.ln_bias   = nn.Parameter(torch.zeros(norm_shape, dtype=torch.float32))

        self.avg_pool = nn.AvgPool3d(kernel_size=pool_kernel_size)
        self.eps = float(eps)

    def forward(self, x):
        # (1) deconvolution
        x = self.conv_transpose(x).contiguous()          # (N,C,D,H,W)

        # (2) fused add-scalar + LayerNorm over last dim (width)
        x = _ops.add_scalar_layernorm_cuda(
            x, float(self.sum_weight.detach()), self.eps
        )

        # affine transformation
        x = x * self.ln_weight + self.ln_bias            # (N,C,D,H,W)

        # (3) average pooling
        x = self.avg_pool(x)

        # (4) fast GELU
        x = _ops.gelu_cuda(x.contiguous())
        return x
