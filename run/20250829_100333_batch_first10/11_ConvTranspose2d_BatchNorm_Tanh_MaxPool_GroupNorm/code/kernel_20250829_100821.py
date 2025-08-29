# <complete ModelNew code>
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ----------------------------------------------------------------------
# CUDA: fused (BatchNorm + Tanh)  followed by  2×2 / stride-2 MaxPool
# ----------------------------------------------------------------------
source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

/* ------------------------------------------------------------------ */
/*  Kernel: BatchNorm (running stats, inference) + Tanh + MaxPool2×2  */
/* ------------------------------------------------------------------ */
__global__ void bn_tanh_maxpool2x2_kernel(const float *__restrict__ inp,
                                          const float *__restrict__ weight,
                                          const float *__restrict__ bias,
                                          const float *__restrict__ mean,
                                          const float *__restrict__ var,
                                          float *__restrict__ out,
                                          int N, int C, int H, int W,
                                          float eps)
{
    const int H_out = H >> 1;   // H / 2
    const int W_out = W >> 1;   // W / 2
    const int total = N * C * H_out * W_out;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < total) {
        /* ------------------------- unravel index ------------------------- */
        int w_out = idx % W_out;
        int h_out = (idx / W_out) % H_out;
        int c     = (idx / (W_out * H_out)) % C;
        int n     =  idx / (W_out * H_out * C);

        const int h_in = h_out << 1;   // *2
        const int w_in = w_out << 1;   // *2
        const int base = ((n * C + c) * H + h_in) * W + w_in;

        /* ---------------------- pre-compute constants -------------------- */
        const float gamma   = weight[c];
        const float beta    = bias[c];
        const float mu      = mean[c];
        const float inv_std = rsqrtf(var[c] + eps);

        /* ----------------------- load & transform ------------------------ */
        float4 v;
        v.x = inp[base];
        v.y = inp[base + 1];
        v.z = inp[base + W];
        v.w = inp[base + W + 1];

        v.x = tanhf(gamma * (v.x - mu) * inv_std + beta);
        v.y = tanhf(gamma * (v.y - mu) * inv_std + beta);
        v.z = tanhf(gamma * (v.z - mu) * inv_std + beta);
        v.w = tanhf(gamma * (v.w - mu) * inv_std + beta);

        /* ------------------------ max-pool reduce ------------------------ */
        float m = v.x;
        if (v.y > m) m = v.y;
        if (v.z > m) m = v.z;
        if (v.w > m) m = v.w;

        out[idx] = m;
        idx += blockDim.x * gridDim.x;
    }
}

/* ---------------------------  Host wrapper  --------------------------- */
torch::Tensor fused_bn_tanh_maxpool2x2_cuda(torch::Tensor input,
                                            torch::Tensor weight,
                                            torch::Tensor bias,
                                            torch::Tensor running_mean,
                                            torch::Tensor running_var,
                                            double  eps)
{
    TORCH_CHECK(input.is_cuda(),        "input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat, "only float32 supported");
    TORCH_CHECK(input.is_contiguous(),  "input must be contiguous");
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);
    TORCH_CHECK((H & 1) == 0 && (W & 1) == 0,
                "H and W must be even for 2×2 / stride-2 max-pool");

    auto output = torch::empty({N, C, H / 2, W / 2}, input.options());

    const int threads = 256;
    const int64_t total = N * C * (H / 2) * (W / 2);
    const int blocks  = (total + threads - 1) / threads;

    bn_tanh_maxpool2x2_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(N), static_cast<int>(C),
        static_cast<int>(H), static_cast<int>(W),
        static_cast<float>(eps));

    return output;
}
"""

cpp_src = r"""
torch::Tensor fused_bn_tanh_maxpool2x2_cuda(torch::Tensor input,
                                            torch::Tensor weight,
                                            torch::Tensor bias,
                                            torch::Tensor running_mean,
                                            torch::Tensor running_var,
                                            double eps);
"""

# --------------------------- build the extension ---------------------------
fused_kernels = load_inline(
    name         = "fused_bn_tanh_maxpool2x2",
    cpp_sources  = cpp_src,
    cuda_sources = source,
    functions    = ["fused_bn_tanh_maxpool2x2_cuda"],
    verbose      = False,
    extra_cuda_cflags=["-O3"],
    extra_cflags=["-O3"]
)

# ----------------------------------------------------------------------
# High-level PyTorch module that uses the fused kernel
# ----------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Optimised model:
      ConvTranspose2d  →  fused (BatchNorm + Tanh + 2×2 MaxPool)  →  GroupNorm
    The fused CUDA kernel eliminates an intermediate tensor and halves the
    global memory traffic compared to the previous two-pass implementation.
    """
    def __init__(self,
                 in_channels, out_channels,
                 kernel_size, stride, padding,
                 groups, num_groups,
                 eps: float = 1e-5):
        super().__init__()
        # first stage: learnable deconvolution
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size,
            stride=stride,
            padding=padding
        )

        # BatchNorm (running stats only — inference) parameters
        self.weight = nn.Parameter(torch.ones(out_channels))
        self.bias   = nn.Parameter(torch.zeros(out_channels))
        self.register_buffer("running_mean", torch.zeros(out_channels))
        self.register_buffer("running_var",  torch.ones(out_channels))
        self.eps = eps

        # final GroupNorm as in original model
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) deconvolution
        x = self.conv_transpose(x)

        # 2) fused BatchNorm + Tanh + 2×2/stride-2 MaxPool
        x = fused_kernels.fused_bn_tanh_maxpool2x2_cuda(
            x.contiguous(),           # ensure memory layout
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            self.eps
        )

        # 3) GroupNorm (unchanged)
        x = self.group_norm(x)
        return x
