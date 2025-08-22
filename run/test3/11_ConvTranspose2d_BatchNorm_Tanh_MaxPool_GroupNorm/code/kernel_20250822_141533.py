import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------------
# CUDA source with custom kernels (NO PYBIND11_MODULE!)
# ------------------------------------------------------------------
source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cfloat>

#define THREADS 256

// ---------------------------------------------------------
// Helper to launch 1-D grids
// ---------------------------------------------------------
inline dim3 GET_BLOCKS(const int64_t N) {
    return dim3((N + THREADS - 1) / THREADS);
}

/* =========================================================
   1) Transposed Convolution (stride=1) – NCHW
        (re-implemented using atomics for correctness)
   ========================================================= */
__global__ void conv_transpose2d_kernel_atomic(
        const float* __restrict__ inp,
        const float* __restrict__ w,
        const float* __restrict__ b,
        float* __restrict__ out,
        int N, int Cin, int Hin, int Win,
        int Cout, int K, int pad,
        int Hout, int Wout) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_inp = (int64_t)N * Cin * Hin * Win;
    if (idx >= total_inp) return;

    int iw  = idx % Win;
    int ih  = (idx / Win) % Hin;
    int ic  = (idx / (Win * Hin)) % Cin;
    int n   = idx / (Win * Hin * Cin);

    float in_val = inp[idx];

    // For each kernel position & output channel: accumulate into output with atomics
    for (int kh = 0; kh < K; ++kh){
        int oh = ih + kh - pad;
        if (oh < 0 || oh >= Hout) continue;
        for (int kw = 0; kw < K; ++kw){
            int ow = iw + kw - pad;
            if (ow < 0 || ow >= Wout) continue;
            for (int oc = 0; oc < Cout; ++oc){
                int64_t w_offset  = (((ic * Cout + oc) * K + kh) * K + kw);
                float    w_val    = w[w_offset];

                int64_t out_offset = (((n * Cout + oc) * Hout + oh) * Wout + ow);
                atomicAdd(&out[out_offset], in_val * w_val);
            }
        }
    }
}

torch::Tensor conv_transpose2d_cuda(torch::Tensor input,
                                    torch::Tensor weight,
                                    torch::Tensor bias,
                                    int padding) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(bias.is_cuda(), "bias must be CUDA");

    const int N   = input.size(0);
    const int Cin = input.size(1);
    const int Hin = input.size(2);
    const int Win = input.size(3);
    const int Cout = weight.size(1);
    const int K = weight.size(2);

    const int Hout = Hin - 1 - 2 * padding + K;
    const int Wout = Win - 1 - 2 * padding + K;

    auto options = input.options();
    auto output  = torch::zeros({N, Cout, Hout, Wout}, options);  // zero-init for atomics

    int64_t total_inp = (int64_t)N * Cin * Hin * Win;
    conv_transpose2d_kernel_atomic<<<GET_BLOCKS(total_inp), THREADS>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),      // bias added after kernel
        output.data_ptr<float>(),
        N, Cin, Hin, Win, Cout, K, padding, Hout, Wout);

    // Add bias
    output = output + bias.view({1, Cout, 1, 1});

    return output;
}

/* =========================================================
   2) BatchNorm (inference) – corrected channel indexing
   ========================================================= */
__global__ void batch_norm_inference_kernel(
        const float* __restrict__ inp,
        const float* __restrict__ gamma,
        const float* __restrict__ beta,
        const float* __restrict__ mean,
        const float* __restrict__ var,
        float* __restrict__ out,
        int64_t total,
        int C,
        int spatial,      // H*W
        float eps) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // Channel index independent of batch
    int c = (idx / spatial) % C;

    float inv_std = rsqrtf(var[c] + eps);
    out[idx] = (inp[idx] - mean[c]) * inv_std * gamma[c] + beta[c];
}

torch::Tensor batch_norm_inference_cuda(torch::Tensor input,
                                        torch::Tensor gamma,
                                        torch::Tensor beta,
                                        torch::Tensor running_mean,
                                        torch::Tensor running_var,
                                        float eps) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int spatial = H * W;

    auto output = torch::empty_like(input);
    int64_t total = input.numel();

    batch_norm_inference_kernel<<<GET_BLOCKS(total), THREADS>>>(
        input.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        output.data_ptr<float>(),
        total, C, spatial, eps);
    return output;
}

/* =========================================================
   3) Element-wise tanh
   ========================================================= */
__global__ void tanh_kernel(const float* x, float* y, int64_t n){
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    y[idx] = tanhf(x[idx]);
}
torch::Tensor tanh_cuda(torch::Tensor input){
    auto output = torch::empty_like(input);
    int64_t n = input.numel();
    tanh_kernel<<<GET_BLOCKS(n), THREADS>>>(input.data_ptr<float>(),
                                            output.data_ptr<float>(), n);
    return output;
}

/* =========================================================
   4) MaxPool2d (kernel=2, stride=2, NCHW)
   ========================================================= */
__global__ void maxpool2d_kernel(
        const float* __restrict__ inp,
        float* __restrict__ out,
        int N, int C,
        int Hin, int Win,
        int Hout, int Wout) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N * C * Hout * Wout;
    if (idx >= total) return;

    int ow = idx % Wout;
    int oh = (idx / Wout) % Hout;
    int c  = (idx / (Wout * Hout)) % C;
    int n  = idx / (Wout * Hout * C);

    int ih0 = oh * 2;
    int iw0 = ow * 2;

    float maxval = -FLT_MAX;
    for (int kh = 0; kh < 2; ++kh){
        for (int kw = 0; kw < 2; ++kw){
            int ih = ih0 + kh;
            int iw = iw0 + kw;
            int in_offset = (((n * C + c) * Hin + ih) * Win + iw);
            maxval = fmaxf(maxval, inp[in_offset]);
        }
    }
    out[idx] = maxval;
}

torch::Tensor max_pool2d_cuda(torch::Tensor input){
    int N = input.size(0);
    int C = input.size(1);
    int Hin = input.size(2);
    int Win = input.size(3);
    int Hout = Hin / 2;
    int Wout = Win / 2;

    auto output = torch::empty({N, C, Hout, Wout}, input.options());
    int64_t total = (int64_t)N * C * Hout * Wout;

    maxpool2d_kernel<<<GET_BLOCKS(total), THREADS>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, Hin, Win, Hout, Wout);
    return output;
}

/* =========================================================
   5) GroupNorm
   ========================================================= */
// First pass: compute mean and var per (N,G)
__global__ void group_stats_kernel(
        const float* __restrict__ x,
        float* __restrict__ mean,
        float* __restrict__ var,
        int N, int C, int H, int W,
        int G, float eps) {

    extern __shared__ float shmem[];          // dynamic shared memory
    float* sh_sum  = shmem;
    float* sh_sum2 = shmem + THREADS;

    int gid = blockIdx.x;                 // Each block handles one (n,g)
    int n  = gid / G;
    int g  = gid % G;
    int channels_per_group = C / G;
    int elements_per_group = channels_per_group * H * W;

    // Local accumulators
    float sum  = 0.f;
    float sum2 = 0.f;

    int c_start = g * channels_per_group;
    for (int c = 0; c < channels_per_group; ++c){
        int ch = c_start + c;
        int base = ((n * C + ch) * H) * W;
        for (int hw = threadIdx.x; hw < H * W; hw += blockDim.x){
            float v = x[base + hw];
            sum  += v;
            sum2 += v * v;
        }
    }
    // Store into shared memory
    sh_sum [threadIdx.x] = sum;
    sh_sum2[threadIdx.x] = sum2;
    __syncthreads();

    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1){
        if (threadIdx.x < s){
            sh_sum [threadIdx.x] += sh_sum [threadIdx.x + s];
            sh_sum2[threadIdx.x] += sh_sum2[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0){
        float m  = sh_sum[0]  / elements_per_group;
        float v  = sh_sum2[0] / elements_per_group - m * m;
        mean[gid] = m;
        var [gid] = v;
    }
}

__global__ void group_norm_apply_kernel(
        const float* __restrict__ x,
        const float* __restrict__ mean,
        const float* __restrict__ var,
        const float* __restrict__ gamma,
        const float* __restrict__ beta,
        float* __restrict__ y,
        int N, int C, int H, int W,
        int G, float eps){

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N * C * H * W;
    if (idx >= total) return;

    int w  = idx % W;
    int h  = (idx / W) % H;
    int c  = (idx / (W * H)) % C;
    int n  = idx / (W * H * C);

    int channels_per_group = C / G;
    int g = c / channels_per_group;
    int gid = n * G + g;

    float inv_std = rsqrtf(var[gid] + eps);
    float xn = (x[idx] - mean[gid]) * inv_std;
    y[idx] = xn * gamma[c] + beta[c];
}

torch::Tensor group_norm_cuda(torch::Tensor input,
                              torch::Tensor gamma,
                              torch::Tensor beta,
                              int groups,
                              float eps){
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    auto mean = torch::empty({N * groups}, input.options());
    auto var  = torch::empty({N * groups}, input.options());

    // stats kernel: one block per (n,g)
    int blocks_stats = N * groups;
    group_stats_kernel<<<blocks_stats, THREADS, THREADS * 2 * sizeof(float)>>>(
        input.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        N, C, H, W, groups, eps);

    auto output = torch::empty_like(input);
    int64_t total = input.numel();
    group_norm_apply_kernel<<<GET_BLOCKS(total), THREADS>>>(
        input.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W, groups, eps);
    return output;
}
"""

# ------------------------------------------------------------------
# C++ prototypes
# ------------------------------------------------------------------
cpp_src = """
torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int padding);
torch::Tensor batch_norm_inference_cuda(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, torch::Tensor running_mean, torch::Tensor running_var, float eps);
torch::Tensor tanh_cuda(torch::Tensor input);
torch::Tensor max_pool2d_cuda(torch::Tensor input);
torch::Tensor group_norm_cuda(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, int groups, float eps);
"""

# ------------------------------------------------------------------
# Build the extension
# ------------------------------------------------------------------
kernels = load_inline(
    name="custom_kernels_fused_fixed",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=[
        "conv_transpose2d_cuda",
        "batch_norm_inference_cuda",
        "tanh_cuda",
        "max_pool2d_cuda",
        "group_norm_cuda",
    ],
    verbose=False,
)

# ------------------------------------------------------------------
# PyTorch Module replacement
# ------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Model that mirrors the original architecture using custom CUDA kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        super().__init__()
        # Parameters for transposed convolution
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, kernel_size, kernel_size))
        self.bias   = nn.Parameter(torch.zeros(out_channels))
        self.padding = padding

        # BatchNorm parameters/buffers
        self.bn_gamma = nn.Parameter(torch.ones(out_channels))
        self.bn_beta  = nn.Parameter(torch.zeros(out_channels))
        self.register_buffer("running_mean", torch.zeros(out_channels))
        self.register_buffer("running_var",  torch.ones(out_channels))
        self.bn_eps = 1e-5

        # GroupNorm parameters
        self.groups = num_groups
        self.gn_gamma = nn.Parameter(torch.ones(out_channels))
        self.gn_beta  = nn.Parameter(torch.zeros(out_channels))
        self.gn_eps   = 1e-5

        # Save references to kernels
        self.kernels = kernels

    def forward(self, x):
        # 1) transposed convolution
        x = self.kernels.conv_transpose2d_cuda(x, self.weight, self.bias, self.padding)
        # 2) batch norm (inference mode using running stats)
        x = self.kernels.batch_norm_inference_cuda(
            x, self.bn_gamma, self.bn_beta, self.running_mean, self.running_var, self.bn_eps
        )
        # 3) tanh activation
        x = self.kernels.tanh_cuda(x)
        # 4) max pool 2d (kernel=2, stride=2)
        x = self.kernels.max_pool2d_cuda(x)
        # 5) group norm
        x = self.kernels.group_norm_cuda(x, self.gn_gamma, self.gn_beta, self.groups, self.gn_eps)
        return x
