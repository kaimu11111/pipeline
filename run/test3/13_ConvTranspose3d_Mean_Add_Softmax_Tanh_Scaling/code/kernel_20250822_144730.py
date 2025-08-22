import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------------
# CUDA source for 3-D transposed convolution (stride = 1, dilation = 1, groups = 1)
#   • input   : (B, Ci, Di, Hi, Wi)
#   • weight  : (Ci, Co,   Kd, Kh, Kw)   – identical layout to PyTorch’s ConvTranspose3d
#   • padding : int  (same pad for D/H/W)
#   • output  : (B, Co, Do, Ho, Wo), where
#                Do = Di + 2*padding - Kd + 1   (stride==1)
#                (similarly for Ho, Wo)
# ------------------------------------------------------------------
source_conv = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// ------------------------------------------------------------------
// GPU kernel: each thread computes one output element (b, co, d, h, w)
// ------------------------------------------------------------------
template <typename scalar_t>
__global__ void conv_transpose3d_kernel(
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ weight,
        scalar_t* __restrict__ output,
        int B, int Ci, int Di, int Hi, int Wi,
        int Co,
        int Kd, int Kh, int Kw,
        int pad,
        int Do, int Ho, int Wo)
{
    int ow = blockIdx.x * blockDim.x + threadIdx.x;          // W  axis
    int oh = blockIdx.y * blockDim.y + threadIdx.y;          // H  axis
    int od = blockIdx.z * blockDim.z + threadIdx.z;          // D  axis
    if (ow >= Wo || oh >= Ho || od >= Do) return;

    // iterate over batch and output channels in outer loops for cache reuse
    for (int b = 0; b < B; ++b)
    {
        for (int co = 0; co < Co; ++co)
        {
            scalar_t acc = 0;
            // convolution sum
            for (int ci = 0; ci < Ci; ++ci)
            {
                for (int kd = 0; kd < Kd; ++kd)
                {
                    int id = od + pad - kd;
                    if (id < 0 || id >= Di) continue;
                    for (int kh = 0; kh < Kh; ++kh)
                    {
                        int ih = oh + pad - kh;
                        if (ih < 0 || ih >= Hi) continue;
                        for (int kw = 0; kw < Kw; ++kw)
                        {
                            int iw = ow + pad - kw;
                            if (iw < 0 || iw >= Wi) continue;

                            // index calculations
                            int input_idx  = (((b * Ci + ci) * Di + id) * Hi + ih) * Wi + iw;
                            int weight_idx = ((((ci * Co + co) * Kd + kd) * Kh + kh) * Kw + kw);
                            acc += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
            int out_idx = (((b * Co + co) * Do + od) * Ho + oh) * Wo + ow;
            output[out_idx] = acc;
        }
    }
}

// ------------------------------------------------------------------
// Host (C++) wrapper
// ------------------------------------------------------------------
torch::Tensor conv_transpose3d_cuda(
        torch::Tensor input,
        torch::Tensor weight,
        int padding)
{
    const int B  = input.size(0);
    const int Ci = input.size(1);
    const int Di = input.size(2);
    const int Hi = input.size(3);
    const int Wi = input.size(4);

    const int Co = weight.size(1);
    const int Kd = weight.size(2);
    const int Kh = weight.size(3);
    const int Kw = weight.size(4);

    const int Do = Di + 2*padding - Kd + 1;
    const int Ho = Hi + 2*padding - Kh + 1;
    const int Wo = Wi + 2*padding - Kw + 1;

    auto output = torch::zeros({B, Co, Do, Ho, Wo}, input.options());

    const dim3 block(8, 4, 2);                                  // 64 threads
    const dim3 grid( (Wo+block.x-1)/block.x,
                     (Ho+block.y-1)/block.y,
                     (Do+block.z-1)/block.z );

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_transpose3d_kernel", ([&] {
        conv_transpose3d_kernel<scalar_t><<<grid, block>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            B, Ci, Di, Hi, Wi,
            Co,
            Kd, Kh, Kw,
            padding,
            Do, Ho, Wo);
    }));
    return output;
}
"""

cpp_conv = "torch::Tensor conv_transpose3d_cuda(torch::Tensor input, torch::Tensor weight, int padding);"

# ------------------------------------------------------------------
# CUDA source for post-processing:
#   mean-over-depth  → +bias → softmax(c) → tanh → scale
#   • input   : (B, C, D, H, W)
#   • bias    : (1, C, 1, 1, 1)    (broadcastable)
#   • scale   : float
#   • output  : (B, C, 1, H, W)
# ------------------------------------------------------------------
source_post = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__global__ void postproc_kernel(
        const scalar_t* __restrict__ in,
        const scalar_t* __restrict__ bias,
        scalar_t* __restrict__ out,
        int B, int C, int D, int H, int W,
        scalar_t scale)
{
    // Each block handles a spatial position (b, h, w)
    int w = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
    if (w >= W || h >= H || b >= B) return;

    // Shared memory for per-channel values
    extern __shared__ scalar_t sdata[];
    scalar_t* vals = sdata;            // C floats

    // ------------------------------------------------------------------
    // 1. Mean over depth  +  bias
    // ------------------------------------------------------------------
    for (int c = threadIdx.x; c < C; c += blockDim.x)
    {
        scalar_t sum = 0;
        for (int d = 0; d < D; ++d)
        {
            int idx = (((b * C + c) * D + d) * H + h) * W + w;
            sum += in[idx];
        }
        scalar_t mean = sum / D;
        vals[c] = mean + bias[c];       // bias has shape (C)
    }
    __syncthreads();

    // ------------------------------------------------------------------
    // 2. Softmax across channels
    //    – compute max
    // ------------------------------------------------------------------
    scalar_t max_val = -1e20;
    for (int c = threadIdx.x; c < C; c += blockDim.x)
        max_val = fmaxf(max_val, vals[c]);
    // reduction within block
    __shared__ scalar_t block_max;
    if (threadIdx.x == 0) block_max = -1e20;
    __syncthreads();
    atomicMax((int*)&block_max, __float_as_int(max_val));
    __syncthreads();
    scalar_t m = block_max;

    // 3. exp & sum
    scalar_t local_sum = 0;
    for (int c = threadIdx.x; c < C; c += blockDim.x)
    {
        vals[c] = expf(vals[c] - m);
        local_sum += vals[c];
    }
    // reduction for sum
    __shared__ scalar_t block_sum;
    if (threadIdx.x == 0) block_sum = 0;
    __syncthreads();
    atomicAdd(&block_sum, local_sum);
    __syncthreads();
    scalar_t s = block_sum + 1e-8;

    // ------------------------------------------------------------------
    // 4. Normalise → tanh → scale
    // ------------------------------------------------------------------
    for (int c = threadIdx.x; c < C; c += blockDim.x)
    {
        scalar_t soft = vals[c] / s;
        scalar_t ta   = tanhf(soft);
        scalar_t res  = ta * scale;

        int out_idx = (((b * C + c) * 1) * H + h) * W + w;   // D dimension collapsed to 1
        out[out_idx] = res;
    }
}

// ------------------------------------------------------------------
torch::Tensor postprocess_cuda(torch::Tensor inp,
                               torch::Tensor bias,
                               float scale)
{
    const int B = inp.size(0);
    const int C = inp.size(1);
    const int D = inp.size(2);
    const int H = inp.size(3);
    const int W = inp.size(4);

    auto out = torch::empty({B, C, 1, H, W}, inp.options());

    dim3 grid(W, H, B);
    int threads = 128;
    size_t shmem = C * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(inp.scalar_type(), "postproc_kernel", ([&] {
        postproc_kernel<scalar_t><<<grid, threads, shmem>>>(
            inp.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            B, C, D, H, W,
            static_cast<scalar_t>(scale) );
    }));
    return out;
}
"""

cpp_post = "torch::Tensor postprocess_cuda(torch::Tensor inp, torch::Tensor bias, float scale);"

# ------------------------------------------------------------------
# Compile kernels
# ------------------------------------------------------------------
conv_cuda = load_inline(name="conv_transpose3d_custom",
                        cpp_sources=cpp_conv,
                        cuda_sources=source_conv,
                        functions=["conv_transpose3d_cuda"],
                        verbose=False)

post_cuda = load_inline(name="postprocess_custom",
                        cpp_sources=cpp_post,
                        cuda_sources=source_post,
                        functions=["postprocess_cuda"],
                        verbose=False)

# ------------------------------------------------------------------
# Model replacement
# ------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Fully custom-kernel version of the reference model
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super().__init__()
        assert stride == 1, "custom kernel supports stride == 1 only"
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        self.padding      = padding
        self.scaling      = float(scaling_factor)

        # trainable weight identical to ConvTranspose3d layout: (Ci, Co, kD, kH, kW)
        weight = torch.empty(in_channels, out_channels,
                             self.kernel_size, self.kernel_size, self.kernel_size)
        nn.init.kaiming_uniform_(weight, a=5**0.5)
        self.weight = nn.Parameter(weight)

        # bias (broadcastable over spatial dims)
        self.bias = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        # x : (B, Ci, D, H, W)
        y  = conv_cuda.conv_transpose3d_cuda(x, self.weight, self.padding)   # (B, Co, D', H', W')
        z  = post_cuda.postprocess_cuda(y, self.bias, self.scaling)          # (B, Co, 1, H', W')
        return z
