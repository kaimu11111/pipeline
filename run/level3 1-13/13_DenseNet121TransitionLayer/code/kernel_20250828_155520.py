import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------------
# Hand-written CUDA kernels + host wrappers
# ---------------------------------------------------------------------------
source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ============================================================
// ReLU → 1×1 Convolution (point-wise)
// ------------------------------------------------------------
template <typename scalar_t>
__global__ void relu_conv1x1_kernel(const scalar_t* __restrict__ input,
                                    const scalar_t* __restrict__ weight,
                                    scalar_t* __restrict__ output,
                                    int N, int C_in, int H, int W, int C_out) {
    const int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * C_out * H * W;
    if (idx >= total) return;

    // Un-flatten index
    const int w  =  idx % W;
    const int h  = (idx / W) % H;
    const int co = (idx / (W * H)) % C_out;
    const int n  =  idx / (W * H * C_out);

    // Pointer offsets
    const scalar_t* weight_ptr = weight + co * C_in;          // (C_out, C_in)
    const scalar_t* in_ptr     = input  + ((n * C_in * H) + h) * W + w;

    scalar_t acc = 0;
    #pragma unroll
    for (int ci = 0; ci < C_in; ++ci) {
        scalar_t v = in_ptr[ci * H * W];
        v = v > 0 ? v : 0;                 // ReLU BEFORE conv
        acc += v * weight_ptr[ci];
    }
    output[idx] = acc;
}

// Host wrapper
torch::Tensor relu_conv1x1_cuda(torch::Tensor input, torch::Tensor weight) {
    TORCH_CHECK(input.is_cuda(),  "input must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(input.dtype() == weight.dtype(), "dtype mismatch");
    TORCH_CHECK(input.dim()  == 4, "input must be NCHW");
    TORCH_CHECK(weight.dim() == 2, "weight must be Cout×Cin");
    TORCH_CHECK(input.size(1) == weight.size(1), "Cin mismatch");

    const int N      = input.size(0);
    const int C_in   = input.size(1);
    const int H      = input.size(2);
    const int W      = input.size(3);
    const int C_out  = weight.size(0);

    auto output = torch::empty({N, C_out, H, W}, input.options());

    const int threads = 256;
    const int total   = N * C_out * H * W;
    const int blocks  = (total + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "relu_conv1x1_cuda", ([&] {
        relu_conv1x1_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N, C_in, H, W, C_out);
    }));
    return output;
}

// ============================================================
// 2×2 Average-Pooling (stride 2)
// ------------------------------------------------------------
template <typename scalar_t>
__global__ void avgpool2x2_kernel(const scalar_t* __restrict__ input,
                                  scalar_t* __restrict__ output,
                                  int N, int C, int H, int W) {
    const int H_out = H >> 1;
    const int W_out = W >> 1;
    const int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * C * H_out * W_out;
    if (idx >= total) return;

    const int w_out =  idx % W_out;
    const int h_out = (idx / W_out) % H_out;
    const int c     = (idx / (W_out * H_out)) % C;
    const int n     =  idx / (W_out * H_out * C);

    const int h_in = h_out << 1;
    const int w_in = w_out << 1;

    const scalar_t* base = input + ((n * C + c) * H + h_in) * W + w_in;

    const scalar_t v00 = base[0];
    const scalar_t v01 = base[1];
    const scalar_t v10 = base[W];
    const scalar_t v11 = base[W + 1];

    output[idx] = static_cast<scalar_t>(0.25) * (v00 + v01 + v10 + v11);
}

// Host wrapper
torch::Tensor avgpool2x2_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(input.dim() == 4, "input must be NCHW");
    TORCH_CHECK((input.size(2) & 1) == 0 && (input.size(3) & 1) == 0,
                "H and W must be even");

    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);
    const int H_out = H >> 1;
    const int W_out = W >> 1;

    auto output = torch::empty({N, C, H_out, W_out}, input.options());

    const int threads = 256;
    const int total   = N * C * H_out * W_out;
    const int blocks  = (total + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "avgpool2x2_cuda", ([&] {
        avgpool2x2_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N, C, H, W);
    }));
    return output;
}
"""

# ---------------------------------------------------------------------------
# Function prototypes required by load_inline
# ---------------------------------------------------------------------------
cpp_src = """
torch::Tensor relu_conv1x1_cuda(torch::Tensor input, torch::Tensor weight);
torch::Tensor avgpool2x2_cuda(torch::Tensor input);
"""

# ---------------------------------------------------------------------------
# Compile / load kernels
# ---------------------------------------------------------------------------
custom_ops = load_inline(
    name        = "fused_pointwise_pool",
    cpp_sources = cpp_src,
    cuda_sources= source,
    functions   = ["relu_conv1x1_cuda", "avgpool2x2_cuda"],
    verbose     = False
)

# ---------------------------------------------------------------------------
# Optimised model using the custom kernels
# ---------------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    BN → ReLU → 1×1 Conv → 2×2 AvgPool
    ReLU is fused INTO the convolution for efficiency.
    """
    def __init__(self, num_input_features: int, num_output_features: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_input_features, affine=True, eps=1e-5)
        self.weight = nn.Parameter(
            torch.empty(num_output_features, num_input_features)
        )
        nn.init.kaiming_uniform_(self.weight, a=(5 ** 0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)                                    # BN
        x = custom_ops.relu_conv1x1_cuda(x, self.weight)  # ReLU → Conv
        x = custom_ops.avgpool2x2_cuda(x)                 # 2×2 AvgPool
        return x
