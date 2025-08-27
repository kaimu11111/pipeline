import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------------
# CUDA / C++  kernel: fuse mean-over-depth + bias + softmax(channel) + tanh + scale
# ---------------------------------------------------------------------------
cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

// ---------------------------------------------------------------------------
// CUDA kernel
// ---------------------------------------------------------------------------
__global__ void fused_kernel(const float* __restrict__ input,
                             const float* __restrict__ bias,
                             float* __restrict__ output,
                             const int B, const int C, const int D,
                             const int H, const int W,
                             const float scaling)
{
    const int idx  = blockIdx.x * blockDim.x + threadIdx.x;
    const int HW   = H * W;
    const int N    = B * HW;          // number of (b,h,w) groups

    if (idx >= N) return;

    // Decode (b,h,w) indices
    const int b  = idx / HW;
    const int hw = idx % HW;
    const int h  = hw / W;
    const int w  = hw % W;

    const int stride_d = H * W;       // distance between consecutive depth slices
    // -----------------------------------------------------------------------
    // 1) find channel-wise maxima after mean&bias for numerical stability
    // -----------------------------------------------------------------------
    float max_val = -FLT_MAX;
    for (int c = 0; c < C; ++c)
    {
        const int base_in = (((b * C + c) * D) * H + h) * W + w;
        float accum = 0.f;
        #pragma unroll
        for (int d = 0; d < D; ++d)
            accum += input[base_in + d * stride_d];
        const float mean_bias = accum / D + bias[c];
        if (mean_bias > max_val)
            max_val = mean_bias;
    }

    // -----------------------------------------------------------------------
    // 2) compute denominator (sum of exp)
    // -----------------------------------------------------------------------
    float exp_sum = 0.f;
    for (int c = 0; c < C; ++c)
    {
        const int base_in = (((b * C + c) * D) * H + h) * W + w;
        float accum = 0.f;
        #pragma unroll
        for (int d = 0; d < D; ++d)
            accum += input[base_in + d * stride_d];
        const float mean_bias = accum / D + bias[c];
        exp_sum += __expf(mean_bias - max_val);
    }
    // Guard against division by zero (shouldn't happen, but be safe)
    exp_sum = exp_sum > 0.f ? exp_sum : 1.f;

    // -----------------------------------------------------------------------
    // 3) final pass – write out: tanh(softmax(mean+bias)) * scaling
    // -----------------------------------------------------------------------
    for (int c = 0; c < C; ++c)
    {
        const int base_in  = (((b * C + c) * D) * H + h) * W + w;
        const int base_out = ((((b * C + c) * 1) * H + h) * W + w); // D==1

        float accum = 0.f;
        #pragma unroll
        for (int d = 0; d < D; ++d)
            accum += input[base_in + d * stride_d];
        const float mean_bias = accum / D + bias[c];

        const float soft = __expf(mean_bias - max_val) / exp_sum;
        output[base_out] = tanhf(soft) * scaling;
    }
}

// ---------------------------------------------------------------------------
// C++/CUDA interface
// ---------------------------------------------------------------------------
torch::Tensor fused_forward_cuda(torch::Tensor input,
                                 torch::Tensor bias,
                                 const float   scaling)
{
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(),  "bias  must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32,
                "only float32 is supported");

    input = input.contiguous();
    bias  = bias.contiguous();

    const int B = input.size(0);
    const int C = input.size(1);
    const int D = input.size(2);
    const int H = input.size(3);
    const int W = input.size(4);

    auto output = torch::empty({B, C, 1, H, W}, input.options());

    const int HW       = H * W;
    const int groups   = B * HW;
    const int threads  = 256;
    const int blocks   = (groups + threads - 1) / threads;

    fused_kernel<<<blocks, threads>>>(input.data_ptr<float>(),
                                      bias.data_ptr<float>(),
                                      output.data_ptr<float>(),
                                      B, C, D, H, W, scaling);

    return output;
}
"""

cpp_decl = "torch::Tensor fused_forward_cuda(torch::Tensor input, torch::Tensor bias, const float scaling);"

# Compile / load the CUDA extension
fused_ops = load_inline(name="fused_depth_softmax_tanh",
                        cpp_sources=cpp_decl,
                        cuda_sources=cuda_src,
                        functions=["fused_forward_cuda"],
                        verbose=False)

# ---------------------------------------------------------------------------
# Optimised model using custom fused CUDA kernel
# ---------------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Optimised version of the reference model.
    Keeps PyTorch's ConvTranspose3d but fuses the subsequent
    mean-over-depth + bias + softmax + tanh + scaling into a single CUDA kernel.
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding,
                 scaling_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels,
            kernel_size, stride=stride, padding=padding)
        # Bias term applied after mean pooling (per channel, broadcasted)
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = float(scaling_factor)

    def forward(self, x):
        x = self.conv_transpose(x)  # (B, C, D, H, W)
        # fused CUDA kernel performs:
        #   mean over depth  -> +bias -> softmax(channel) -> tanh -> scale
        x = fused_ops.fused_forward_cuda(x, self.bias, self.scaling_factor)
        return x
