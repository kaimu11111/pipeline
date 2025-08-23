import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused Mish -> add_value -> Hardtanh -> scale
source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Fused kernel: out = scale * clamp(mish(x) + add_value, -1, 1)
__global__ void fused_mish_add_hardtanh_scale_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float add_value,
    float scale,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        // mish(val) = val * tanh(ln(1 + exp(val)))
        float sp = logf(1.0f + expf(val));
        float mish_val = val * tanhf(sp);
        float added = mish_val + add_value;
        // clamp to [-1, 1]
        if (added > 1.0f) added = 1.0f;
        if (added < -1.0f) added = -1.0f;
        // scale
        output[idx] = added * scale;
    }
}

torch::Tensor fused_mish_add_hardtanh_scale_cuda(
    torch::Tensor input,
    float add_value,
    float scale
) {
    auto output = torch::zeros_like(input);
    int size = input.numel();

    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;

    fused_mish_add_hardtanh_scale_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        add_value,
        scale,
        size
    );
    return output;
}
""";

cpp_src = r"""
torch::Tensor fused_mish_add_hardtanh_scale_cuda(
    torch::Tensor input,
    float add_value,
    float scale
);
"""

fused_mish_add_hardtanh_scale = load_inline(
    name="fused_mish_add_hardtanh_scale",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["fused_mish_add_hardtanh_scale_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a transposed convolution, then applies
    fused Mish->add->Hardtanh->scale with a custom CUDA kernel.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        add_value,
        scale
    ):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding
        )
        self.add_value = add_value
        self.scale = scale
        self.fused_op = fused_mish_add_hardtanh_scale

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_op.fused_mish_add_hardtanh_scale_cuda(x, self.add_value, self.scale)
        return x
