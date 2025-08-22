import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------------------
# CUDA / C++ source
# ---------------------------------------------------------------------------------
source = r"""
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>   // <-- added for AT_DISPATCH_* macros
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

/* -------------------------------------------------------------------------- */
/*                                CUDA helpers                                */
/* -------------------------------------------------------------------------- */
#define CUDA_1D_KERNEL_LOOP(i, n)                                             \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; (i) < (n);              \
       i += blockDim.x * gridDim.x)

/* ---------------------------- type converters ----------------------------- */
__device__ inline float to_float(const float v)  { return v; }
__device__ inline float to_float(const double v) { return static_cast<float>(v); }
__device__ inline float to_float(const __half v) { return __half2float(v); }

__device__ inline float to_float(const at::BFloat16 v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __bfloat162float(v);
#else
    return static_cast<float>(v);   // fallback – acceptable for our purposes
#endif
}

template <typename scalar_t>
__device__ inline scalar_t from_float(const float v);

template <>
__device__ inline float from_float<float>(const float v) {
    return v;
}

template <>
__device__ inline __half from_float<__half>(const float v) {
    return __float2half_rn(v);
}

template <>
__device__ inline at::BFloat16 from_float<at::BFloat16>(const float v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __float2bfloat16(v);
#else
    return static_cast<at::BFloat16>(v);
#endif
}

/* -------------------------------------------------------------------------- */
/*                         Fused activation (templated)                       */
/* -------------------------------------------------------------------------- */
template <typename scalar_t>
__global__ void activation_kernel(const scalar_t* __restrict__ in,
                                  scalar_t*       __restrict__ out,
                                  const int total,
                                  const float add_value,
                                  const float scale) {
    CUDA_1D_KERNEL_LOOP(idx, total) {
        float x     = to_float(in[idx]);

        /* Mish activation: x * tanh(ln(1 + exp(x))) */
        float sp    = logf(1.0f + expf(x));
        float mish  = x * tanhf(sp);

        /* Add, scale, then Hardtanh clamp */
        float val   = (mish + add_value) * scale;
        val         = fminf(fmaxf(val, -1.0f), 1.0f);

        out[idx] = from_float<scalar_t>(val);
    }
}

/* -------------------------------------------------------------------------- */
/*                           Host wrapper function                            */
/* -------------------------------------------------------------------------- */
torch::Tensor conv_transpose2d_fused_cuda(torch::Tensor input,
                                          torch::Tensor weight,
                                          int  stride,
                                          int  padding,
                                          int  output_padding,
                                          float add_value,
                                          float scale) {
    /* Ensure contiguous CUDA tensors */
    input  = input.contiguous();
    weight = weight.contiguous();

    /* ---- Forward transposed convolution (uses PyTorch optimised kernel) ---- */
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

    /* Allocate output tensor with same properties as conv_out */
    torch::Tensor output = torch::empty_like(conv_out);

    const int total   = conv_out.numel();
    const int threads = 256;
    const int blocks  = (total + threads - 1) / threads;

    /* ---------------- Dispatch according to tensor dtype ------------------- */
    AT_DISPATCH_FLOATING_TYPES_AND2(at::Half, at::BFloat16,
                                    conv_out.scalar_type(),
                                    "activation_kernel_launch", ([&] {
        activation_kernel<scalar_t><<<blocks, threads>>>(
            conv_out.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            total,
            add_value,
            scale
        );
    }));

    /* Propagate kernel launch errors */
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("CUDA kernel failed : ", cudaGetErrorString(err));
    }

    return output;
}
"""

# --------------------------- C++ function prototypes ---------------------------
cpp_src = r"""
torch::Tensor conv_transpose2d_fused_cuda(torch::Tensor input,
                                          torch::Tensor weight,
                                          int  stride,
                                          int  padding,
                                          int  output_padding,
                                          float add_value,
                                          float scale);
"""

# ------------------------------ load extension ---------------------------------
conv_transpose2d_fused = load_inline(
    name="conv_transpose2d_fused_v3_fix",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["conv_transpose2d_fused_cuda"],
    verbose=False,
)

# ---------------------------------------------------------------------------------
#                                 Model wrapper
# ---------------------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Transposed convolution + Mish + add constant + scale + HardTanh ([-1, 1]).
    Implemented with a custom CUDA fused kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, output_padding,
                 add_value, scale):
        super().__init__()
        # ConvTranspose2d expects (in_channels, out_channels, kH, kW)
        self.weight = nn.Parameter(torch.empty(
            in_channels, out_channels, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=(5 ** 0.5))

        self.stride          = int(stride)
        self.padding         = int(padding)
        self.output_padding  = int(output_padding)
        self.add_value       = float(add_value)
        self.scale           = float(scale)

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
