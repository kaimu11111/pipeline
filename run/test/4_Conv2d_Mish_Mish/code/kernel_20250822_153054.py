import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------
# CUDA source: grouped 2-D VALID convolution fused with Mish activation
# ---------------------------------------------------------------------
source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>

__device__ __forceinline__ float mish_act(float x) {
    const float threshold = 20.f;
    float softplus;
    if (x > threshold)
        softplus = x;                              // ln(1+e^x) ~ x (large pos)
    else if (x < -threshold)
        softplus = __expf(x);                      // ln(1+e^x) ~ e^x (large neg)
    else
        softplus = log1pf(__expf(x));              // stable default
    return x * tanhf(softplus);
}

/*
 * Each thread computes exactly one output element (VALID padding, stride=1):
 *   out(n, co, h_out, w_out)
 *
 * Grid:
 *   blockDim  = (TX, TY, 1)
 *   gridDim.x = ceil(W_out / TX)
 *   gridDim.y = ceil(H_out / TY)
 *   gridDim.z = N * C_out                (each (n,co) pair)
 */
template<int TX, int TY>
__global__ void fused_conv2d_valid_mish_kernel(
        const float *__restrict__ input,     // (N, C_in,  H,     W)
        const float *__restrict__ weight,    // (C_out, C_in/G, K_h, K_w) – contiguous
        const float *__restrict__ bias,      // (C_out) or nullptr
        float       *__restrict__ output,    // (N, C_out, H_out, W_out)
        int N, int C_in, int H, int W,
        int C_out,
        int K_h, int K_w,
        int H_out, int W_out,
        int groups) {

    const int w_out = blockIdx.x * TX + threadIdx.x;
    const int h_out = blockIdx.y * TY + threadIdx.y;
    const int nc    = blockIdx.z;               // 0 … N*C_out-1
    const int n     = nc / C_out;
    const int co    = nc - n * C_out;

    if (w_out >= W_out || h_out >= H_out) return;

    const int C_out_per_g = C_out / groups;
    const int C_in_per_g  = C_in  / groups;
    const int g           = co / C_out_per_g;   // group index

    float acc = (bias != nullptr) ? bias[co] : 0.0f;

#pragma unroll
    for (int ci_local = 0; ci_local < C_in_per_g; ++ci_local) {
        const int ci = g * C_in_per_g + ci_local;   // global input-channel id

#pragma unroll
        for (int kh = 0; kh < K_h; ++kh) {
            const int h_in = h_out + kh;            // VALID padding

#pragma unroll
            for (int kw = 0; kw < K_w; ++kw) {
                const int w_in = w_out + kw;        // VALID padding

                /* 64-bit indexing avoids overflow for large tensors */
                const size_t input_idx =
                    (((size_t)n * C_in + ci) * H + h_in) * W + w_in;

                const size_t weight_idx =
                    ((((size_t)co * C_in_per_g + ci_local) * K_h + kh) * K_w) + kw;

                acc += input[input_idx] * weight[weight_idx];
            }
        }
    }

    // Mish activation
    acc = mish_act(acc);

    const size_t out_idx =
        (((size_t)n * C_out + co) * H_out + h_out) * W_out + w_out;
    output[out_idx] = acc;
}

// ---------------------------------------------------------------------
// Host wrapper
// ---------------------------------------------------------------------
torch::Tensor fused_conv2d_valid_mish_cuda(torch::Tensor input,
                                           torch::Tensor weight,
                                           torch::Tensor bias,
                                           int groups) {
    TORCH_CHECK(input.is_cuda(),  "input must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(input.dtype()  == torch::kFloat32, "only float32 supported");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "only float32 supported");
    TORCH_CHECK(input.dim()  == 4 && weight.dim() == 4, "expect NCHW / OIHW");

    const int N      = input.size(0);
    const int C_in   = input.size(1);
    const int H      = input.size(2);
    const int W      = input.size(3);
    const int C_out  = weight.size(0);
    const int K_h    = weight.size(2);
    const int K_w    = weight.size(3);

    TORCH_CHECK(C_in % groups == 0 && C_out % groups == 0,
                "C_in and C_out must be divisible by groups");

    const int H_out = H - K_h + 1;
    const int W_out = W - K_w + 1;
    TORCH_CHECK(H_out > 0 && W_out > 0, "kernel larger than input");

    auto out = torch::empty({N, C_out, H_out, W_out},
                            input.options());

    constexpr int TX = 16;
    constexpr int TY = 16;
    dim3 block(TX, TY, 1);
    dim3 grid( (W_out + TX - 1) / TX,
               (H_out + TY - 1) / TY,
               N * C_out );

    fused_conv2d_valid_mish_kernel<TX, TY><<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        out.data_ptr<float>(),
        N, C_in, H, W,
        C_out,
        K_h, K_w,
        H_out, W_out,
        groups
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(err));

    return out;
}
"""

# ---------------------------------------------------------------------
# C++ prototypes exposed to Python
# ---------------------------------------------------------------------
cpp_src = """
torch::Tensor fused_conv2d_valid_mish_cuda(torch::Tensor input,
                                           torch::Tensor weight,
                                           torch::Tensor bias,
                                           int groups);
"""

# ---------------------------------------------------------------------
# Compile / load the CUDA extension
# ---------------------------------------------------------------------
fused_ops = load_inline(
    name="fused_conv2d_valid_mish",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["fused_conv2d_valid_mish_cuda"],
    verbose=False,
)

# ---------------------------------------------------------------------
# PyTorch Module wrapping the fused kernel
# ---------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Convolution (VALID padding) + Mish fused into a single custom CUDA kernel.
    Supports arbitrary groups.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups=1, bias=True):
        super().__init__()
        assert in_channels % groups == 0 and out_channels % groups == 0
        self.groups = groups
        if isinstance(kernel_size, int):
            kh = kw = kernel_size
        else:
            kh, kw = kernel_size
        self.kernel_size = (kh, kw)

        self.weight = nn.Parameter(
            torch.empty(out_channels,
                        in_channels // groups,
                        kh, kw)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        # Weight initialization identical to nn.Conv2d default
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if bias:
            fan_in = in_channels * kh * kw / groups
            bound = 1 / fan_in ** 0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return fused_ops.fused_conv2d_valid_mish_cuda(
            x, self.weight,
            self.bias if self.bias is not None else torch.Tensor(),  # dummy if None
            self.groups
        )
