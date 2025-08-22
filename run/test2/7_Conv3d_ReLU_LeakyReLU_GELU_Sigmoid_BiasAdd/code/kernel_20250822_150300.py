import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ----------------------------------------------------------------------------- 
# CUDA kernels and C++ wrappers
# -----------------------------------------------------------------------------
source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_1D_KERNEL_LOOP(i, n)                                     \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);        \
       i += blockDim.x * gridDim.x)

/* ----------------------------- 3-D Convolution ---------------------------- */
__global__ void conv3d_forward_kernel(
        const float* __restrict__ x,           // [N,C_in,D,H,W]
        const float* __restrict__ w,           // [C_out,C_in,K,K,K]
        float* __restrict__ y,                 // [N,C_out,D_out,H_out,W_out]
        const int N,
        const int C_in,
        const int D, const int H, const int W,
        const int C_out,
        const int K,
        const int D_out, const int H_out, const int W_out) {

    const int out_elems = N * C_out * D_out * H_out * W_out;
    CUDA_1D_KERNEL_LOOP(idx, out_elems) {
        int tmp = idx;
        const int w_out_idx = tmp % W_out; tmp /= W_out;
        const int h_out_idx = tmp % H_out; tmp /= H_out;
        const int d_out_idx = tmp % D_out; tmp /= D_out;
        const int co = tmp % C_out; tmp /= C_out;
        const int n  = tmp;

        float sum = 0.f;
        for (int ci = 0; ci < C_in; ++ci) {
            for (int kd = 0; kd < K; ++kd) {
                int d_in = d_out_idx + kd;
                for (int kh = 0; kh < K; ++kh) {
                    int h_in = h_out_idx + kh;
                    for (int kw = 0; kw < K; ++kw) {
                        int w_in = w_out_idx + kw;
                        int x_idx = (((n * C_in + ci) * D + d_in) * H + h_in) * W + w_in;
                        int w_idx = ((((co * C_in + ci) * K + kd) * K + kh) * K) + kw;
                        sum += x[x_idx] * w[w_idx];
                    }
                }
            }
        }
        int y_idx = (((n * C_out + co) * D_out + d_out_idx) * H_out + h_out_idx) * W_out + w_out_idx;
        y[y_idx] = sum;
    }
}

torch::Tensor conv3d_forward_cuda(torch::Tensor x, torch::Tensor w) {
    TORCH_CHECK(x.is_cuda(), "input must be CUDA");
    TORCH_CHECK(w.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "only float32 supported");
    TORCH_CHECK(w.scalar_type() == torch::kFloat32, "only float32 supported");

    const int N = x.size(0);
    const int C_in = x.size(1);
    const int D = x.size(2);
    const int H = x.size(3);
    const int W = x.size(4);

    const int C_out = w.size(0);
    const int K = w.size(2);  // assume cubic kernel and w shape [C_out,C_in,K,K,K]

    const int D_out = D - K + 1;
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    auto y = torch::empty({N, C_out, D_out, H_out, W_out}, x.options());

    const int threads = 256;
    const int blocks = (N * C_out * D_out * H_out * W_out + threads - 1) / threads;

    conv3d_forward_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        y.data_ptr<float>(),
        N, C_in, D, H, W, C_out, K,
        D_out, H_out, W_out
    );

    return y;
}

/* ------------------------- Element-wise Point Operators -------------------- */
template <typename Op>
__global__ void unary_elemwise_kernel(float* __restrict__ x, const float* __restrict__ in,
                                      int64_t numel, Op op) {
    CUDA_1D_KERNEL_LOOP(i, numel) {
        x[i] = op(in[i]);
    }
}

struct ReluFunctor {
    __device__ float operator()(float v) const { return v > 0.f ? v : 0.f; }
};
struct LeakyReluFunctor {
    float neg;
    __host__ __device__ LeakyReluFunctor(float n):neg(n){}
    __device__ float operator()(float v) const { return v > 0.f ? v : v * neg; }
};
struct GeluFunctor {
    __device__ float operator()(float v) const {
        const float k0 = 0.7978845608f;  // sqrt(2/pi)
        const float k1 = 0.044715f;
        return 0.5f * v * (1.f + tanhf(k0 * (v + k1 * v * v * v)));
    }
};
struct SigmoidFunctor {
    __device__ float operator()(float v) const { return 1.f / (1.f + expf(-v)); }
};

torch::Tensor relu_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "input must be CUDA");
    auto out = torch::empty_like(x);
    const int64_t numel = x.numel();
    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;
    unary_elemwise_kernel<<<blocks, threads>>>(out.data_ptr<float>(),
                                               x.data_ptr<float>(),
                                               numel,
                                               ReluFunctor());
    return out;
}

torch::Tensor leaky_relu_cuda(torch::Tensor x, float negative_slope) {
    TORCH_CHECK(x.is_cuda(), "input must be CUDA");
    auto out = torch::empty_like(x);
    const int64_t numel = x.numel();
    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;
    unary_elemwise_kernel<<<blocks, threads>>>(out.data_ptr<float>(),
                                               x.data_ptr<float>(),
                                               numel,
                                               LeakyReluFunctor(negative_slope));
    return out;
}

torch::Tensor gelu_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "input must be CUDA");
    auto out = torch::empty_like(x);
    const int64_t numel = x.numel();
    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;
    unary_elemwise_kernel<<<blocks, threads>>>(out.data_ptr<float>(),
                                               x.data_ptr<float>(),
                                               numel,
                                               GeluFunctor());
    return out;
}

torch::Tensor sigmoid_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "input must be CUDA");
    auto out = torch::empty_like(x);
    const int64_t numel = x.numel();
    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;
    unary_elemwise_kernel<<<blocks, threads>>>(out.data_ptr<float>(),
                                               x.data_ptr<float>(),
                                               numel,
                                               SigmoidFunctor());
    return out;
}

/* --------------------------- Add bias (C-wise) ----------------------------- */
__global__ void add_bias_kernel(
        float* __restrict__ y,
        const float* __restrict__ bias,   // [C_out]
        const int N, const int C, const int D, const int H, const int W) {
    const int total = N * C * D * H * W;
    CUDA_1D_KERNEL_LOOP(idx, total) {
        int tmp = idx / (D * H * W);
        int c = tmp % C;
        y[idx] += bias[c];
    }
}

torch::Tensor add_bias_cuda(torch::Tensor x, torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda() && bias.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(bias.dim() == 1, "bias should be 1-D (C_out)");
    auto y = x.clone();
    const int N = y.size(0);
    const int C = y.size(1);
    const int D = y.size(2);
    const int H = y.size(3);
    const int W = y.size(4);

    const int threads = 256;
    const int blocks = (N * C * D * H * W + threads - 1) / threads;

    add_bias_kernel<<<blocks, threads>>>(y.data_ptr<float>(),
                                         bias.data_ptr<float>(),
                                         N, C, D, H, W);
    return y;
}
"""

# ----------------------------------------------------------------------------- 
# C++ interface prototypes
# -----------------------------------------------------------------------------
cpp_src = """
torch::Tensor conv3d_forward_cuda(torch::Tensor x, torch::Tensor w);
torch::Tensor relu_cuda(torch::Tensor x);
torch::Tensor leaky_relu_cuda(torch::Tensor x, float negative_slope);
torch::Tensor gelu_cuda(torch::Tensor x);
torch::Tensor sigmoid_cuda(torch::Tensor x);
torch::Tensor add_bias_cuda(torch::Tensor x, torch::Tensor bias);
"""

# ----------------------------------------------------------------------------- 
# Compile and load
# -----------------------------------------------------------------------------
ops = load_inline(
    name="fused_ops",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=[
        "conv3d_forward_cuda",
        "relu_cuda",
        "leaky_relu_cuda",
        "gelu_cuda",
        "sigmoid_cuda",
        "add_bias_cuda",
    ],
    verbose=False,
)

# ----------------------------------------------------------------------------- 
# PyTorch module using the custom kernels
# -----------------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    3D convolution followed by a sequence of activation functions and bias add,
    implemented with custom CUDA kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, negative_slope=0.01):
        super(ModelNew, self).__init__()
        weight_shape = (out_channels, in_channels, kernel_size, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.randn(weight_shape))
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.kernel_size = kernel_size
        self.negative_slope = negative_slope

    def forward(self, x):
        w = self.weight.to(x.device)
        b = self.bias.to(x.device)

        x = ops.conv3d_forward_cuda(x, w)
        x = ops.relu_cuda(x)
        x = ops.leaky_relu_cuda(x, float(self.negative_slope))
        x = ops.gelu_cuda(x)
        x = ops.sigmoid_cuda(x)
        x = ops.add_bias_cuda(x, b)
        return x
