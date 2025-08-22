import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------------
# C++/CUDA source: use ATen conv3d (stride=1, dilation=1)
# ------------------------------------------------------------------
source_conv = r"""
#include <torch/extension.h>

torch::Tensor conv3d_cuda(torch::Tensor input,
                          torch::Tensor weight,
                          int padding)
{
    TORCH_CHECK(input.is_cuda() && weight.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(input.scalar_type() == weight.scalar_type(), "dtype mismatch");
    /* stride  = (1,1,1)
       padding = (p,p,p)
       dilation/output_pad/groups left at default */
    return at::conv3d(
            input,                         // (B, Ci, D, H, W)
            weight,                        // (Co, Ci, kD, kH, kW)
            c10::optional<torch::Tensor>(),// no bias
            {1,1,1},                       // stride
            {padding,padding,padding},     // padding
            {1,1,1});                      // dilation
}
"""

cpp_conv = "torch::Tensor conv3d_cuda(torch::Tensor input, torch::Tensor weight, int padding);"

# ------------------------------------------------------------------
# CUDA source for post-processing kernel (unchanged – fully custom)
# ------------------------------------------------------------------
source_post = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

template <typename scalar_t>
__global__ void postproc_kernel(
        const scalar_t* __restrict__ in,
        const scalar_t* __restrict__ bias,
        scalar_t* __restrict__ out,
        int B, int C, int D, int H, int W,
        scalar_t scale)
{
    /* Each block handles one (b, h, w); threads span C dimension */
    int w = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
    if (w >= W || h >= H || b >= B) return;

    extern __shared__ unsigned char shared_mem[];
    scalar_t* shm_vals = reinterpret_cast<scalar_t*>(shared_mem);      // C
    scalar_t* shm_red  = shm_vals + C;                                 // blockDim.x

    /* 1. Mean over depth + bias */
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        scalar_t sum = 0;
        for (int d = 0; d < D; ++d) {
            int idx = (((b * C + c) * D + d) * H + h) * W + w;
            sum += in[idx];
        }
        shm_vals[c] = sum / static_cast<scalar_t>(D) + bias[c];
    }
    __syncthreads();

    /* 2. Softmax across channels */
    scalar_t thread_max = -FLT_MAX;
    for (int c = threadIdx.x; c < C; c += blockDim.x)
        thread_max = fmaxf(thread_max, shm_vals[c]);

    shm_red[threadIdx.x] = thread_max;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            shm_red[threadIdx.x] = fmaxf(shm_red[threadIdx.x], shm_red[threadIdx.x + stride]);
        __syncthreads();
    }
    scalar_t m = shm_red[0];
    __syncthreads();

    scalar_t thread_sum = 0;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        shm_vals[c] = expf(shm_vals[c] - m);
        thread_sum += shm_vals[c];
    }
    shm_red[threadIdx.x] = thread_sum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            shm_red[threadIdx.x] += shm_red[threadIdx.x + stride];
        __syncthreads();
    }
    scalar_t s = shm_red[0] + 1e-8f;
    __syncthreads();

    /* 3. Normalise → tanh → scale */
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        scalar_t soft = shm_vals[c] / s;
        scalar_t res  = tanhf(soft) * scale;
        int out_idx = (((b * C + c) * 1) * H + h) * W + w;
        out[out_idx] = res;
    }
}

torch::Tensor postprocess_cuda(torch::Tensor inp,
                               torch::Tensor bias,
                               float scale)
{
    TORCH_CHECK(inp.is_cuda() && bias.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(inp.scalar_type() == at::kFloat, "kernel implemented for float only");

    const int B = inp.size(0);
    const int C = inp.size(1);
    const int D = inp.size(2);
    const int H = inp.size(3);
    const int W = inp.size(4);

    auto out = torch::empty({B, C, 1, H, W}, inp.options());

    dim3 grid(W, H, B);
    int threads = 128;
    size_t shmem = (static_cast<size_t>(C) + threads) * sizeof(float);

    postproc_kernel<float><<<grid, threads, shmem>>>(
        inp.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        B, C, D, H, W,
        scale);

    return out;
}
"""

cpp_post = "torch::Tensor postprocess_cuda(torch::Tensor inp, torch::Tensor bias, float scale);"

# ------------------------------------------------------------------
# Compile kernels
# ------------------------------------------------------------------
conv_cuda = load_inline(name="conv3d_custom",
                        cpp_sources=cpp_conv,
                        cuda_sources=source_conv,
                        functions=["conv3d_cuda"],
                        verbose=False)

post_cuda = load_inline(name="postprocess_custom",
                        cpp_sources=cpp_post,
                        cuda_sources=source_post,
                        functions=["postprocess_cuda"],
                        verbose=False)

# ------------------------------------------------------------------
# Model wrapper
# ------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Fully custom-kernel version of the reference model.
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding, scaling_factor):
        super().__init__()
        assert stride == 1, "custom kernel supports stride == 1 only"
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        self.padding      = padding
        self.scaling      = float(scaling_factor)

        # Conv3d weight layout: (Co, Ci, kD, kH, kW)
        weight = torch.empty(out_channels, in_channels,
                             self.kernel_size, self.kernel_size, self.kernel_size,
                             device='cuda', dtype=torch.float32)
        nn.init.kaiming_uniform_(weight, a=5 ** 0.5)
        self.weight = nn.Parameter(weight)

        # Bias (broadcastable over spatial dims): shape (Co)
        bias = torch.randn(out_channels, device='cuda', dtype=torch.float32)
        self.bias = nn.Parameter(bias)

    def forward(self, x):
        # x : (B, Ci, D, H, W) – float32 CUDA tensor
        y = conv_cuda.conv3d_cuda(x, self.weight, self.padding)   # (B, Co, D, H, W)
        z = post_cuda.postprocess_cuda(y, self.bias, self.scaling)          # (B, Co, 1, H, W)
        return z
