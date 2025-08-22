import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -------------------------------- CUDA & C++ source ---------------------------------
source = r"""
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

/* -------------------------------- Helpers ----------------------------------- */
#define CUDA_1D_KERNEL_LOOP(i, n)                                \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; (i) < (n); \
       i += blockDim.x * gridDim.x)

/* ---------------------------- Fused activation ------------------------------ */
__global__ void activation_kernel(const float* __restrict__ in,
                                  float* __restrict__ out,
                                  const int total,
                                  const float add_value,
                                  const float scale) {
    CUDA_1D_KERNEL_LOOP(idx, total) {
        float x   = in[idx];

        /* Mish activation: x * tanh(ln(1 + exp(x))) */
        float sp  = logf(1.0f + expf(x));
        float mish = x * tanhf(sp);

        /* Add constant, clamp (Hardtanh), and scale */
        float val = mish + add_value;
        val       = fminf(fmaxf(val, -1.0f), 1.0f);  // Hardtanh [-1, 1]
        out[idx]  = val * scale;
    }
}

/* ---------------------- Host wrapper (C++ / CUDA) --------------------------- */
torch::Tensor conv_transpose2d_fused_cuda(torch::Tensor input,
                                          torch::Tensor weight,
                                          int stride,
                                          int padding,
                                          int output_padding,
                                          float add_value,
                                          float scale) {
    /* Ensure contiguous tensors on CUDA */
    input  = input.contiguous();
    weight = weight.contiguous();

    /* --- Call highly-optimised PyTorch transposed-convolution --- */
    torch::Tensor conv_out = at::conv_transpose2d(
        input,
        weight,
        /*bias=*/c10::nullopt,
        /*stride=*/{stride, stride},
        /*padding=*/{padding, padding},
        /*output_padding=*/{output_padding, output_padding},
        /*groups=*/1,
        /*dilation=*/{1, 1}
    );

    /* --- Allocate output tensor --- */
    torch::Tensor output = torch::empty_like(conv_out);

    /* --- Launch fused element-wise kernel --- */
    const int total   = conv_out.numel();
    const int threads = 256;
    const int blocks  = (total + threads - 1) / threads;

    activation_kernel<<<blocks, threads>>>(
        conv_out.data_ptr<float>(),
        output.data_ptr<float>(),
        total,
        add_value,
        scale
    );

    /* --- Propagate any kernel launch errors --- */
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel failed : ", cudaGetErrorString(err));
    }

    return output;
}
"""

# --------------------------- C++ function prototypes ---------------------------
cpp_src = """
torch::Tensor conv_transpose2d_fused_cuda(torch::Tensor input,
                                          torch::Tensor weight,
                                          int stride,
                                          int padding,
                                          int output_padding,
                                          float add_value,
                                          float scale);
"""

# ------------------------------- load extension --------------------------------
conv_transpose2d_fused = load_inline(
    name="conv_transpose2d_fused",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["conv_transpose2d_fused_cuda"],
    verbose=False,
)

# --------------------------------- ModelNew -----------------------------------
class ModelNew(nn.Module):
    """
    Transposed convolution + Mish + add constant + Hardtanh + scale fused.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 output_padding, add_value, scale):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels,
                                               kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.stride = int(stride)
        self.padding = int(padding)
        self.output_padding = int(output_padding)
        self.add_value = float(add_value)
        self.scale = float(scale)

    def forward(self, x):
        return conv_transpose2d_fused.conv_transpose2d_fused_cuda(
            x,
            self.weight,
            self.stride,
            self.padding,
            self.output_padding,
            self.add_value,
            self.scale
        )
