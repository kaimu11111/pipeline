import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------
# CUDA sources
# ---------------------------------------------------------------------
source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT32(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")

// ---------------------------------------------------------------------
// 1. Naive NCHW conv2d (stride=1, padding=0, dilation=1)
// ---------------------------------------------------------------------
__global__ void conv2d_forward_kernel(
    const float* __restrict__ input,      // [N, Ci, H, W]
    const float* __restrict__ weight,     // [Co, Ci, K, K]
    const float* __restrict__ bias,       // [Co] or nullptr
    float* __restrict__ output,           // [N, Co, Ho, Wo]
    int N, int Ci, int H, int W,
    int Co, int K, int Ho, int Wo)
{
    // blockIdx.z -> combined (n, co)
    int nc = blockIdx.z;
    int n  = nc / Co;
    int co = nc % Co;

    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;

    if (h_out >= Ho || w_out >= Wo) return;

    const int in_hw_stride   = W;
    const int in_ch_stride   = H * W;
    const int in_batch_stride= Ci * H * W;

    const int wt_hw_stride   = K;
    const int wt_ch_stride   = K * K;
    const int wt_out_stride  = Ci * K * K;

    float acc = bias != nullptr ? bias[co] : 0.0f;

    for (int ci = 0; ci < Ci; ++ci) {
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                int h_in = h_out + kh;
                int w_in = w_out + kw;

                float val = input[n * in_batch_stride +
                                   ci * in_ch_stride +
                                   h_in * in_hw_stride +
                                   w_in];

                float wt  = weight[co * wt_out_stride +
                                   ci * wt_ch_stride +
                                   kh * wt_hw_stride +
                                   kw];

                acc += val * wt;
            }
        }
    }

    output[n * (Co * Ho * Wo) +
           co * (Ho * Wo) +
           h_out * Wo +
           w_out] = acc;
}

// ---------------------------------------------------------------------
// 2. InstanceNorm + divide kernel helpers
//    (per-sample, per-channel mean/var over spatial dims)
// ---------------------------------------------------------------------
template<int BLOCK_SIZE>
__global__ void calc_mean_invstd_kernel(
        const float* __restrict__ x,   // [N, C, H, W]
        float* __restrict__ mean,      // [N*C]
        float* __restrict__ invstd,    // [N*C]
        int H, int W, float eps)
{
    // Each block handles one (n,c)
    int nc = blockIdx.x;
    int S  = H * W;

    // Initialise shared memory
    __shared__ float shm[2 * BLOCK_SIZE];  // sum and sumsq

    float local_sum   = 0.0f;
    float local_sumsq = 0.0f;

    // Strided loop over spatial elements
    for (int idx = threadIdx.x; idx < S; idx += BLOCK_SIZE) {
        int h = idx / W;
        int w = idx % W;
        float val = x[nc * S + h * W + w];
        local_sum   += val;
        local_sumsq += val * val;
    }
    shm[threadIdx.x]                 = local_sum;
    shm[BLOCK_SIZE + threadIdx.x]    = local_sumsq;
    __syncthreads();

    // Reduction
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shm[threadIdx.x]              += shm[threadIdx.x + stride];
            shm[BLOCK_SIZE + threadIdx.x] += shm[BLOCK_SIZE + threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float sum   = shm[0];
        float sumsq = shm[BLOCK_SIZE];
        float mu    = sum / S;
        float var   = sumsq / S - mu * mu;
        mean[nc]    = mu;
        invstd[nc]  = rsqrtf(var + eps);
    }
}

// Normalize, divide kernel
__global__ void norm_div_kernel(
        const float* __restrict__ x,   // [N, C, H, W]
        const float* __restrict__ mean,    // [N*C]
        const float* __restrict__ invstd,  // [N*C]
        float* __restrict__ y,
        int H, int W, float divide_by)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   // element index in flattened tensor
    int total = gridDim.x * blockDim.x;

    int S = H * W;
    int N_C = gridDim.y;

    for (int linear = idx; linear < N_C * S; linear += total) {
        int nc = linear / S;
        int hw = linear % S;
        float val = x[linear];
        float mu  = mean[nc];
        float inv = invstd[nc];
        y[linear] = ((val - mu) * inv) / divide_by;
    }
}

// ---------------------------------------------------------------------
// C++/CUDA interface helpers
// ---------------------------------------------------------------------
torch::Tensor conv2d_forward_cuda(torch::Tensor input,
                                  torch::Tensor weight,
                                  torch::Tensor bias) {
    CHECK_CUDA(input); CHECK_CONTIGUOUS(input); CHECK_FLOAT32(input);
    CHECK_CUDA(weight); CHECK_CONTIGUOUS(weight); CHECK_FLOAT32(weight);
    if (bias.defined()) { CHECK_CUDA(bias); CHECK_CONTIGUOUS(bias); CHECK_FLOAT32(bias); }

    auto N  = input.size(0);
    auto Ci = input.size(1);
    auto H  = input.size(2);
    auto W  = input.size(3);
    auto Co = weight.size(0);
    auto K  = weight.size(2);

    int Ho = H - K + 1;
    int Wo = W - K + 1;

    auto output = torch::empty({N, Co, Ho, Wo}, input.options());

    dim3 block(16, 16);
    dim3 grid((Wo + block.x - 1) / block.x,
              (Ho + block.y - 1) / block.y,
              N * Co);

    conv2d_forward_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, Ci, H, W, Co, K, Ho, Wo);

    return output;
}

torch::Tensor instance_norm_divide_cuda(torch::Tensor input,
                                        double divide_by,
                                        double eps=1e-5) {
    CHECK_CUDA(input); CHECK_CONTIGUOUS(input); CHECK_FLOAT32(input);

    auto sizes = input.sizes();
    int N = sizes[0];
    int C = sizes[1];
    int H = sizes[2];
    int W = sizes[3];

    auto output = torch::empty_like(input);
    auto mean   = torch::empty({N * C}, input.options());
    auto invstd = torch::empty({N * C}, input.options());

    // Kernel 1: mean & invstd
    const int BLOCK_SIZE = 256;
    const int shared_mem = 2 * BLOCK_SIZE * sizeof(float);
    calc_mean_invstd_kernel<BLOCK_SIZE><<<N * C, BLOCK_SIZE, shared_mem>>>(
        input.data_ptr<float>(),
        mean.data_ptr<float>(),
        invstd.data_ptr<float>(),
        H, W, static_cast<float>(eps));

    // Kernel 2: normalize & divide
    int S = H * W;
    int threads = 256;
    int blocks = (S + threads - 1) / threads;
    dim3 grid_nd(blocks, N * C);   // y-dim encodes nc
    norm_div_kernel<<<grid_nd, threads>>>(
        input.data_ptr<float>(),
        mean.data_ptr<float>(),
        invstd.data_ptr<float>(),
        output.data_ptr<float>(),
        H, W, static_cast<float>(divide_by));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_forward_cuda", &conv2d_forward_cuda, "Naive Conv2d forward (CUDA)");
    m.def("instance_norm_divide_cuda", &instance_norm_divide_cuda, "InstanceNorm + divide (CUDA)");
}
"""

# ---------------------------------------------------------------------
# C++ function prototypes
# ---------------------------------------------------------------------
cpp_src = """
torch::Tensor conv2d_forward_cuda(torch::Tensor input,
                                  torch::Tensor weight,
                                  torch::Tensor bias);
torch::Tensor instance_norm_divide_cuda(torch::Tensor input,
                                        double divide_by,
                                        double eps=1e-5);
"""

# ---------------------------------------------------------------------
# compilation
# ---------------------------------------------------------------------
kernels = load_inline(
    name="fused_conv_instnorm",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=[
        "conv2d_forward_cuda",
        "instance_norm_divide_cuda",
    ],
    verbose=False,
)

# ---------------------------------------------------------------------
# Model replacement
# ---------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Optimised model with hand-written CUDA kernels replacing
    Conv2d, InstanceNorm2d, and division by constant.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super().__init__()
        # Parameters equivalent to nn.Conv2d with bias=True
        weight = torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        nn.init.kaiming_uniform_(weight, a=5**0.5)
        bias = torch.empty(out_channels)
        fan_in = in_channels * kernel_size * kernel_size
        bound = 1 / fan_in**0.5
        nn.init.uniform_(bias, -bound, bound)
        self.weight = nn.Parameter(weight)
        self.bias   = nn.Parameter(bias)
        self.divide_by = float(divide_by)
        self.kernel_size = kernel_size

    def forward(self, x):
        x = kernels.conv2d_forward_cuda(x.contiguous(),
                                        self.weight.contiguous(),
                                        self.bias.contiguous())
        x = kernels.instance_norm_divide_cuda(x.contiguous(),
                                              self.divide_by)
        return x
