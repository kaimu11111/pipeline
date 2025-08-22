import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------
# CUDA source implementing a fused 2D convolution followed by two Mish
# activations:  y = mish(mish(conv2d(x, w) + b))
# ---------------------------------------------------------------------
source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>

__device__ __forceinline__ float mish_act(float x) {
    // mish(x) = x * tanh(ln(1 + exp(x)))
    return x * tanhf(log1pf(expf(x)));
}

/*
 * Each thread computes exactly one output element corresponding to:
 *   (n, co, h_out, w_out)
 *
 * Grid configuration:
 *   blockDim  = (THREADS_X, THREADS_Y, 1)
 *   gridDim.x = ceil(W_out / THREADS_X)
 *   gridDim.y = ceil(H_out / THREADS_Y)
 *   gridDim.z = N * C_out
 *
 * Memory layout is assumed to be contiguous NCHW.
 */
template<int THREADS_X, int THREADS_Y>
__global__ void fused_conv2d_mish_kernel(
        const float *__restrict__ input,   // (N, C_in, H, W)
        const float *__restrict__ weight,  // (C_out, C_in, K_h, K_w)
        const float *__restrict__ bias,    // (C_out) or nullptr
        float *__restrict__ output,        // (N, C_out, H_out, W_out)
        int N, int C_in, int H, int W,
        int C_out,
        int K_h, int K_w,
        int H_out, int W_out) {

    const int w_out = blockIdx.x * THREADS_X + threadIdx.x;
    const int h_out = blockIdx.y * THREADS_Y + threadIdx.y;
    const int nc   = blockIdx.z;          // 0 .. N*C_out-1
    const int n    = nc / C_out;
    const int co   = nc - n * C_out;

    if (w_out >= W_out || h_out >= H_out) return;

    float acc = bias != nullptr ? bias[co] : 0.0f;

#pragma unroll
    for (int ci = 0; ci < C_in; ++ci) {
#pragma unroll
        for (int kh = 0; kh < K_h; ++kh) {
#pragma unroll
            for (int kw = 0; kw < K_w; ++kw) {
                const int h_in = h_out + kh;
                const int w_in = w_out + kw;

                const int input_idx =
                    ((n * C_in + ci) * H + h_in) * W + w_in;           // NCHW
                const int weight_idx =
                    (((co * C_in + ci) * K_h + kh) * K_w) + kw;        // OIHW

                acc += input[input_idx] * weight[weight_idx];
            }
        }
    }

    // Apply Mish twice
    acc = mish_act(acc);
    acc = mish_act(acc);

    const int out_idx =
        ((n * C_out + co) * H_out + h_out) * W_out + w_out;            // NCHW
    output[out_idx] = acc;
}

// ---------------------------------------------------------------------
// Host wrapper
// ---------------------------------------------------------------------
torch::Tensor fused_conv2d_mish_cuda(torch::Tensor input,
                                     torch::Tensor weight,
                                     torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(),  "input must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(input.dtype()  == torch::kFloat32, "only float32 supported");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "only float32 supported");
    TORCH_CHECK(input.dim()  == 4 && weight.dim() == 4, "NCHW / OIHW");

    const int N      = input.size(0);
    const int C_in   = input.size(1);
    const int H      = input.size(2);
    const int W      = input.size(3);
    const int C_out  = weight.size(0);
    const int K_h    = weight.size(2);
    const int K_w    = weight.size(3);
    const int H_out  = H - K_h + 1;
    const int W_out  = W - K_w + 1;

    auto options = torch::TensorOptions()
                       .dtype(input.dtype())
                       .device(input.device());
    auto output = torch::empty({N, C_out, H_out, W_out}, options);

    constexpr int THREADS_X = 16;
    constexpr int THREADS_Y = 16;
    const dim3 block(THREADS_X, THREADS_Y, 1);
    const dim3 grid( (W_out + THREADS_X - 1) / THREADS_X,
                     (H_out + THREADS_Y - 1) / THREADS_Y,
                     N * C_out );

    fused_conv2d_mish_kernel<THREADS_X, THREADS_Y><<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C_in, H, W,
        C_out,
        K_h, K_w,
        H_out, W_out
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(err));

    return output;
}
"""

# ---------------------------------------------------------------------
# C++ prototypes exposed to Python
# ---------------------------------------------------------------------
cpp_src = """
torch::Tensor fused_conv2d_mish_cuda(torch::Tensor input,
                                     torch::Tensor weight,
                                     torch::Tensor bias);
"""

# ---------------------------------------------------------------------
# Compile / load the CUDA extension
# ---------------------------------------------------------------------
fused_ops = load_inline(
    name="fused_conv2d_mish",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["fused_conv2d_mish_cuda"],
    verbose=False,
)

# ---------------------------------------------------------------------
# PyTorch Module wrapping the fused kernel
# ---------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Re-implementation of the original Model, with convolution + double-Mish
    fused into a single custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias   = nn.Parameter(torch.empty(out_channels))
        # Weight initialization identical to nn.Conv2d default
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = in_channels * kernel_size * kernel_size
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return fused_ops.fused_conv2d_mish_cuda(x, self.weight, self.bias)
