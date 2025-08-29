import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------------
# Hand-written CUDA kernels
# ---------------------------------------------------------------------------
cuda_sources = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ------------------------------------------------------------
// 1×1 Convolution + ReLU (point-wise) kernel
// ------------------------------------------------------------
template <typename scalar_t>
__global__ void conv1x1_relu_kernel(const scalar_t* __restrict__ input,
                                    const scalar_t* __restrict__ weight,
                                    scalar_t* __restrict__ output,
                                    int N, int C_in, int H, int W, int C_out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * C_out * H * W;
    if (idx >= total) return;

    // Un-flatten the linear index
    int w =  idx % W;
    int h = (idx / W) % H;
    int co = (idx / (W * H)) % C_out;
    int n  =  idx / (W * H * C_out);

    // Pointer offsets
    const scalar_t* weight_ptr = weight + co * C_in;          // (C_out, C_in)
    const scalar_t* in_ptr     = input  + ((n * C_in * H) + h) * W + w;

    scalar_t acc = 0;
    #pragma unroll
    for (int ci = 0; ci < C_in; ++ci) {
        acc += in_ptr[ci * H * W] * weight_ptr[ci];
    }
    // Fused ReLU
    acc = acc > 0 ? acc : 0;
    output[idx] = acc;
}

// Host wrapper
torch::Tensor conv1x1_relu_cuda(torch::Tensor input, torch::Tensor weight) {
    TORCH_CHECK(input.is_cuda(),  "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == weight.dtype(), "input/weight dtype mismatch");
    TORCH_CHECK(input.dim()  == 4, "input must be NCHW");
    TORCH_CHECK(weight.dim() == 2, "weight must be Cout x Cin");
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

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv1x1_relu_cuda", ([&] {
        conv1x1_relu_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N, C_in, H, W, C_out);
    }));
    return output;
}

// ------------------------------------------------------------
// 2×2 Average-Pooling (stride 2) kernel
// ------------------------------------------------------------
template <typename scalar_t>
__global__ void avgpool2x2_kernel(const scalar_t* __restrict__ input,
                                  scalar_t* __restrict__ output,
                                  int N, int C, int H, int W) {
    const int H_out = H >> 1;  // divide by 2
    const int W_out = W >> 1;
    const int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * C * H_out * W_out;
    if (idx >= total) return;

    int w_out =  idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int c     = (idx / (W_out * H_out)) % C;
    int n     =  idx / (W_out * H_out * C);

    // Map to top-left corner of the 2×2 window in the input
    const int h_in = h_out << 1;
    const int w_in = w_out << 1;

    const scalar_t* base = input + ((n * C + c) * H + h_in) * W + w_in;

    scalar_t v00 = base[0];
    scalar_t v01 = base[1];
    scalar_t v10 = base[W];
    scalar_t v11 = base[W + 1];

    output[idx] = static_cast<scalar_t>(0.25) * (v00 + v01 + v10 + v11);
}

// Host wrapper
torch::Tensor avgpool2x2_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(),  "input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "input must be NCHW");
    TORCH_CHECK((input.size(2) & 1) == 0 && (input.size(3) & 1) == 0,
                "H and W must be even for 2x2 avg-pool");

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

// Declarations that PyTorch needs
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv1x1_relu_cuda", &conv1x1_relu_cuda, "1x1 Convolution + ReLU (CUDA)");
    m.def("avgpool2x2_cuda",   &avgpool2x2_cuda,   "2x2 AvgPool stride-2 (CUDA)");
}
"""

cpp_declarations = """
torch::Tensor conv1x1_relu_cuda(torch::Tensor input, torch::Tensor weight);
torch::Tensor avgpool2x2_cuda(torch::Tensor input);
"""

# Compile/Load the kernels
custom_ops = load_inline(
    name         = "fused_pointwise_pool",
    cpp_sources  = cpp_declarations,
    cuda_sources = cuda_sources,
    functions    = ["conv1x1_relu_cuda", "avgpool2x2_cuda"],
    verbose      = False
)

# ---------------------------------------------------------------------------
# Optimised model
# ---------------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Optimised version of the original model using custom CUDA kernels:
        BN -> (1×1 Conv + ReLU) -> 2×2 AvgPool (stride 2)
    The convolution and activation are fused, and pooling is custom.
    """
    def __init__(self, num_input_features: int, num_output_features: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_input_features, affine=True, eps=1e-5)
        # 1×1 convolution weights (no bias, as in the original)
        self.weight = nn.Parameter(
            torch.empty(num_output_features, num_input_features)
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # BatchNorm (uses cuDNN / native kernel)
        x = self.bn(x)
        # Fused 1×1 convolution + ReLU
        x = custom_ops.conv1x1_relu_cuda(x, self.weight)
        # 2×2 average pooling (stride 2)
        x = custom_ops.avgpool2x2_cuda(x)
        return x


# ---------------------------------------------------------------------------
# Convenience factory to match original API
# ---------------------------------------------------------------------------
import math
def get_inputs():
    # These dimensions should match the original sample sizes
    batch_size = 32
    num_input_features = 16
    height, width = 128, 128
    return [torch.rand(batch_size, num_input_features, height, width, device="cuda")]

def get_init_inputs():
    # Align with the original signature
    return [16, 32]
