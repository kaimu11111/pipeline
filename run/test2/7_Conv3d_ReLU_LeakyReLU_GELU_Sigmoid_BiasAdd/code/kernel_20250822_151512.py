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
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);    \
       i += static_cast<int64_t>(blockDim.x) * gridDim.x)

/* ----------------------------- 3-D Convolution (valid) --------------------- */
__global__ void conv3d_forward_valid_kernel(
        const float* __restrict__ x,           // [N,C_in,D,H,W]
        const float* __restrict__ w,           // [C_out,C_in,K,K,K]
        float* __restrict__ y,                 // [N,C_out,Dv,Hv,Wv]
        const int64_t N,
        const int64_t C_in,
        const int64_t D, const int64_t H, const int64_t W,
        const int64_t C_out,
        const int64_t K,
        const int64_t Dv, const int64_t Hv, const int64_t Wv) {

    const int64_t out_elems = N * C_out * Dv * Hv * Wv;
    CUDA_1D_KERNEL_LOOP(idx, out_elems) {
        int64_t tmp = idx;
        const int64_t w_out = tmp % Wv; tmp /= Wv;
        const int64_t h_out = tmp % Hv; tmp /= Hv;
        const int64_t d_out = tmp % Dv; tmp /= Dv;
        const int64_t co    = tmp % C_out; tmp /= C_out;
        const int64_t n     = tmp;

        float sum = 0.f;
        for (int64_t ci = 0; ci < C_in; ++ci) {
            for (int64_t kd = 0; kd < K; ++kd) {
                int64_t d_in = d_out + kd;
                for (int64_t kh = 0; kh < K; ++kh) {
                    int64_t h_in = h_out + kh;
                    for (int64_t kw = 0; kw < K; ++kw) {
                        int64_t w_in = w_out + kw;

                        int64_t x_idx = ((((n * C_in + ci) * D + d_in) * H + h_in) * W + w_in);
                        int64_t w_idx = (((((co * C_in + ci) * K + kd) * K + kh) * K) + kw);
                        sum += __ldg(x + x_idx) * __ldg(w + w_idx);
                    }
                }
            }
        }
        int64_t y_idx = ((((n * C_out + co) * Dv + d_out) * Hv + h_out) * Wv + w_out);
        y[y_idx] = sum;
    }
}

torch::Tensor conv3d_forward_cuda(torch::Tensor x, torch::Tensor w) {
    TORCH_CHECK(x.is_cuda(), "input must be CUDA");
    TORCH_CHECK(w.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "only float32 supported");
    TORCH_CHECK(w.scalar_type() == torch::kFloat32, "only float32 supported");

    const int64_t N = x.size(0);
    const int64_t C_in = x.size(1);
    const int64_t D = x.size(2);
    const int64_t H = x.size(3);
    const int64_t W = x.size(4);

    const int64_t C_out = w.size(0);
    const int64_t K = w.size(2);  // assume cubic kernel and w shape [C_out,C_in,K,K,K]

    const int64_t Dv = D - K + 1;
    const int64_t Hv = H - K + 1;
    const int64_t Wv = W - K + 1;
    TORCH_CHECK(Dv > 0 && Hv > 0 && Wv > 0, "kernel too large for input");

    auto y = torch::empty({N, C_out, Dv, Hv, Wv}, x.options());

    const int threads = 256;
    const int64_t blocks = (N * C_out * Dv * Hv * Wv + threads - 1) / threads;

    conv3d_forward_valid_kernel<<<static_cast<int>(blocks), threads>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        y.data_ptr<float>(),
        N, C_in, D, H, W, C_out, K, Dv, Hv, Wv
    );

    return y;
}

/* ------------------------- Element-wise Point Operators -------------------- */
template <typename Op>
__global__ void unary_elemwise_kernel(float* __restrict__ out,
                                      const float* __restrict__ in,
                                      int64_t numel, Op op) {
    CUDA_1D_KERNEL_LOOP(i, numel) {
        out[i] = op(in[i]);
    }
}

struct ReluFunctor {
    __device__ float operator()(float v) const { return v > 0.f ? v : 0.f; }
};
struct LeakyReluFunctor {
    float neg;
    __host__ __device__ explicit LeakyReluFunctor(float n):neg(n){}
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
    const int64_t blocks = (numel + threads - 1) / threads;
    unary_elemwise_kernel<<<static_cast<int>(blocks), threads>>>(out.data_ptr<float>(),
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
    const int64_t blocks = (numel + threads - 1) / threads;
    unary_elemwise_kernel<<<static_cast<int>(blocks), threads>>>(out.data_ptr<float>(),
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
    const int64_t blocks = (numel + threads - 1) / threads;
    unary_elemwise_kernel<<<static_cast<int>(blocks), threads>>>(out.data_ptr<float>(),
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
    const int64_t blocks = (numel + threads - 1) / threads;
    unary_elemwise_kernel<<<static_cast<int>(blocks), threads>>>(out.data_ptr<float>(),
                                                                 x.data_ptr<float>(),
                                                                 numel,
                                                                 SigmoidFunctor());
    return out;
}

/* --------------------------- Add bias (C-wise) ----------------------------- */
__global__ void add_bias_kernel(
        float* __restrict__ y,
        const float* __restrict__ bias,   // [C_out]
        const int64_t N, const int64_t C, const int64_t D,
        const int64_t H, const int64_t W) {
    const int64_t total = N * C * D * H * W;
    CUDA_1D_KERNEL_LOOP(idx, total) {
        int64_t tmp = idx / (D * H * W);
        int64_t c = tmp % C;
        y[idx] += bias[c];
    }
}

torch::Tensor add_bias_cuda(torch::Tensor x, torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda() && bias.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(bias.dim() == 1, "bias should be 1-D (C_out)");
    auto y = x.clone();
    const int64_t N = y.size(0);
    const int64_t C = y.size(1);
    const int64_t D = y.size(2);
    const int64_t H = y.size(3);
    const int64_t W = y.size(4);

    const int threads = 256;
    const int64_t blocks = (N * C * D * H * W + threads - 1) / threads;

    add_bias_kernel<<<static_cast<int>(blocks), threads>>>(
        y.data_ptr<float>(),
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
    name="fused_ops_fixed",
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
    3-D convolution (valid) followed by activations and bias add.
    """
    def __init__(self, in_channels, out_channels, kernel_size, negative_slope=0.01):
        super(ModelNew, self).__init__()

        if isinstance(kernel_size, tuple):
            assert len(kernel_size) == 3, "kernel_size tuple must be of length 3"
            assert kernel_size[0] == kernel_size[1] == kernel_size[2], \
                "only cubic kernels are supported"
            K = kernel_size[0]
        else:
            K = int(kernel_size)

        weight_shape = (out_channels, in_channels, K, K, K)
        self.weight = nn.Parameter(torch.randn(weight_shape))
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.negative_slope = float(negative_slope)

    def forward(self, x):
        y = ops.conv3d_forward_cuda(x, self.weight.to(x.device).contiguous())
        y = ops.relu_cuda(y)
        y = ops.leaky_relu_cuda(y, self.negative_slope)
        y = ops.gelu_cuda(y)
        y = ops.sigmoid_cuda(y)
        y = ops.add_bias_cuda(y, self.bias.to(x.device).contiguous())
        return y
