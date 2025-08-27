import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------------
# Inline CUDA kernels for fast clamping and channel–wise scaling.
# ---------------------------------------------------------------------------
cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

/* ---------------- Element-wise clamp ---------------- */
__global__ void clamp_kernel(const float* __restrict__ in,
                             float* __restrict__ out,
                             float clamp_min,
                             float clamp_max,
                             int n_elem)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elem) return;

    float v = in[idx];
    v = fminf(fmaxf(v, clamp_min), clamp_max);
    out[idx] = v;
}

/* -------- Channel-wise scaling (broadcast over B, D, H, W) -------- */
__global__ void scale_kernel(const float* __restrict__ in,
                             const float* __restrict__ scale,
                             float* __restrict__ out,
                             int channels,
                             int spatial,          // D*H*W
                             int n_elem)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elem) return;

    int c = (idx / spatial) % channels;      // channel index
    float s = scale[c];
    out[idx] = in[idx] * s;
}

/* ------------------ C++/CUDA wrappers ------------------ */
torch::Tensor clamp_cuda(torch::Tensor input,
                         float clamp_min,
                         float clamp_max)
{
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "only float32 supported");

    const int n_elem = input.numel();
    auto out = torch::empty_like(input);

    const int threads = 256;
    const int blocks  = (n_elem + threads - 1) / threads;

    clamp_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        out.data_ptr<float>(),
        clamp_min,
        clamp_max,
        n_elem
    );

    return out;
}

torch::Tensor scale_cuda(torch::Tensor input,
                         torch::Tensor scale)
{
    TORCH_CHECK(input.is_cuda() && scale.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32 &&
                scale.dtype() == torch::kFloat32, "only float32 supported");

    const int B = input.size(0);
    const int C = input.size(1);
    const int D = input.size(2);
    const int H = input.size(3);
    const int W = input.size(4);
    const int spatial = D * H * W;
    const int n_elem  = B * C * spatial;

    auto out = torch::empty_like(input);

    const int threads = 256;
    const int blocks  = (n_elem + threads - 1) / threads;

    scale_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        scale.data_ptr<float>(),
        out.data_ptr<float>(),
        C,
        spatial,
        n_elem
    );

    return out;
}
"""

cpp_src = """
torch::Tensor clamp_cuda(torch::Tensor input,
                         float clamp_min,
                         float clamp_max);

torch::Tensor scale_cuda(torch::Tensor input,
                         torch::Tensor scale);
"""

cuda_ops = load_inline(
    name="custom_clamp_scale",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["clamp_cuda", "scale_cuda"],
    verbose=False,
)

# ---------------------------------------------------------------------------
# Optimised PyTorch model using the custom CUDA kernels.
# ---------------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    AvgPool3D ➔ ConvTranspose3D ➔ Clamp (CUDA) ➔ Spatial Softmax ➔
    Channel-wise Scale (CUDA)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, output_padding, pool_kernel_size,
                 clamp_min, clamp_max):
        super(ModelNew, self).__init__()

        self.avg_pool = nn.AvgPool3d(pool_kernel_size)

        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding
        )

        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

        # Learnable per-channel scale (broadcast over B, D, H, W)
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))

        # Bind CUDA wrappers
        self._clamp_cuda = cuda_ops.clamp_cuda
        self._scale_cuda = cuda_ops.scale_cuda

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv_transpose(x)

        # Fast clamp (CUDA)
        x = self._clamp_cuda(x, self.clamp_min, self.clamp_max)

        # Spatial softmax
        b, c, d, h, w = x.shape
        x = x.view(b, c, -1)
        x = torch.softmax(x, dim=2)
        x = x.view(b, c, d, h, w)

        # Channel-wise scaling (CUDA)
        scale_1d = self.scale.view(-1)  # shape: (C,)
        x = self._scale_cuda(x, scale_1d)

        return x


# ---------------------------------------------------------------------------
# Helpers to match the original API.
# ---------------------------------------------------------------------------
batch_size = 16
in_channels = 16
out_channels = 32
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
pool_kernel_size = 2
clamp_min = 0.0
clamp_max = 1.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width, device="cuda")]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding,
            output_padding, pool_kernel_size, clamp_min, clamp_max]
