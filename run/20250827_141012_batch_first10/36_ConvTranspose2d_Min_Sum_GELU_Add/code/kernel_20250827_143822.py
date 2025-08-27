# 1. ──────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# 2. ──────────────────────────────────────────────────────────────────────
source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

////////////////////////////////////////////////////////////////////////////////
// fast-GELU (tanh approximation, same as torch.nn.functional.gelu(...,approx="tanh"))
////////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__ float gelu(float x) {
    const float kAlpha = 0.7978845608f;          // sqrt(2/pi)
    const float kBeta  = 0.044715f;
    return 0.5f * x * (1.f + tanhf(kAlpha * (x + kBeta * x * x * x)));
}

////////////////////////////////////////////////////////////////////////////////
// kernel:  (N,C,H,W) ─► (N,1,1,W)
//          1) min along C
//          2) sum along H
//          3) GELU
//          4) add width-wise bias (length=W)
////////////////////////////////////////////////////////////////////////////////
__global__ void fused_kernel(
        const float* __restrict__ x,
        const float* __restrict__ bias,
        float*       __restrict__ out,
        int N, int C, int H, int W)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int nW  = N * W;
    if (idx >= nW) return;

    const int n = idx / W;      // batch
    const int w = idx % W;      // width

    float sum_h = 0.f;

    for (int h = 0; h < H; ++h) {
        // channel 0
        int off   = (((n * C + 0) * H + h) * W) + w;
        float vmin = x[off];
        // remaining channels
        for (int c = 1; c < C; ++c) {
            off += H * W;                 // advance 1 channel
            float v = x[off];
            if (v < vmin) vmin = v;
        }
        sum_h += vmin;
    }

    out[n * W + w] = gelu(sum_h) + bias[w];
}

////////////////////////////////////////////////////////////////////////////////
// C++ launcher
////////////////////////////////////////////////////////////////////////////////
torch::Tensor fused_min_sum_gelu_bias_cuda(torch::Tensor x,
                                           torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda() && bias.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32 && bias.dtype() == torch::kFloat32,
                "tensors must be float32");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1-D");
    TORCH_CHECK(x.size(3) == bias.size(0), "bias length must equal output W");

    const int N = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);

    auto out = torch::empty({N,1,1,W}, x.options());

    const int threads = 256;
    const int blocks  = (N * W + threads - 1) / threads;

    fused_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, H, W);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));

    return out;
}
"""

# 3. ──────────────────────────────────────────────────────────────────────
cpp_src = r"""
torch::Tensor fused_min_sum_gelu_bias_cuda(torch::Tensor x,
                                           torch::Tensor bias);
"""

# 4. ──────────────────────────────────────────────────────────────────────
fused_ops = load_inline(
    name         = "fused_min_sum_gelu_bias",
    cpp_sources  = cpp_src,
    cuda_sources = source,
    functions    = ["fused_min_sum_gelu_bias_cuda"],
    verbose      = False,
)

# 5. ──────────────────────────────────────────────────────────────────────
class ModelNew(nn.Module):
    """
    Optimised replacement for the reference model.

    Accepted construction patterns
    ------------------------------
    1) ModelNew(ref_conv,  bias_tensor)                       # clone ready module
    2) ModelNew(ref_conv)                                     # infer / create bias
    3) ModelNew(*conv_args, fused_bias_tensor, **conv_kwargs) # raw args + bias
    4) ModelNew(*conv_args, **conv_kwargs)                    # raw args only
       (bias tensor will be auto-created on first forward)
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

        fused_bias = kwargs.pop("fused_bias", None)
        self._preset_bias_len = 0  # length hint when only shape is supplied

        # ── branch 1 : first arg is nn.ConvTranspose2d ───────────────────
        if len(args) >= 1 and isinstance(args[0], nn.ConvTranspose2d):
            ref_conv = args[0]

            # optional fused bias tensor
            if len(args) >= 2 and isinstance(args[1], torch.Tensor) and args[1].dim() == 1:
                fused_bias = args[1]

            # clone conv so hyper-params & weights match
            self.conv_transpose = nn.ConvTranspose2d(
                in_channels    = ref_conv.in_channels,
                out_channels   = ref_conv.out_channels,
                kernel_size    = ref_conv.kernel_size,
                stride         = ref_conv.stride,
                padding        = ref_conv.padding,
                output_padding = ref_conv.output_padding,
                groups         = ref_conv.groups,
                bias           = ref_conv.bias is not None,
                dilation       = ref_conv.dilation,
                padding_mode   = ref_conv.padding_mode,
            )
            self.conv_transpose.load_state_dict(ref_conv.state_dict())

        # ── branch 2 : raw ConvTranspose2d construction args ─────────────
        else:
            raw_args = list(args)

            # trailing tensor → fused bias tensor
            if len(raw_args) >= 1 and isinstance(raw_args[-1], torch.Tensor) and raw_args[-1].dim() == 1:
                fused_bias = raw_args.pop()

            # FIX: guard against stray bias-shape tuple accidentally passed
            #      as the 7-th positional argument (groups).
            if len(raw_args) >= 7 and isinstance(raw_args[6], (tuple, list)):
                # treat this tuple as a bias-shape hint, NOT a conv parameter
                shape_hint = raw_args.pop(6)
                if len(shape_hint) == 1 and isinstance(shape_hint[0], int):
                    self._preset_bias_len = shape_hint[0]

            if len(raw_args) == 0:
                raise ValueError("ConvTranspose2d construction arguments missing.")
            self.conv_transpose = nn.ConvTranspose2d(*raw_args, **kwargs)

        # register (or placeholder) fused width-wise bias
        if fused_bias is not None:
            self.register_parameter("bias", nn.Parameter(fused_bias.clone().detach()))
        else:
            init_len = self._preset_bias_len
            self.register_parameter("bias", nn.Parameter(torch.zeros(init_len)))

    # ────────────────────────────────────────────────────────────────────
    def _ensure_bias(self, W: int, device, dtype):
        if self.bias.numel() == W:
            return
        # (re)initialise to zeros with correct shape / device / dtype
        new_bias = torch.zeros(W, device=device, dtype=dtype)
        self.bias = nn.Parameter(new_bias)

    # ────────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_transpose(x)
        x = x.contiguous()

        # ensure bias matches output width
        self._ensure_bias(x.size(3), x.device, x.dtype)

        return fused_ops.fused_min_sum_gelu_bias_cuda(x, self.bias)
