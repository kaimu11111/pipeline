import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------------
# Build the fused CUDA kernel (Mish → add → Hardtanh → scale)
# ------------------------------------------------------------------
cpp_src = r"""
#include <torch/extension.h>
torch::Tensor fused_forward_cuda(torch::Tensor input, double add_val, double scale_val);
"""

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float mish(float x) {
    // Numerically-stable softplus: log(1+e^x)
    float sp = logf(1.0f + expf(-fabsf(x))) + fmaxf(x, 0.0f);
    return x * tanhf(sp);
}

__global__ void fused_kernel(const float* __restrict__ in,
                             float* __restrict__ out,
                             int64_t size,
                             float add_val,
                             float scale_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float y = mish(in[idx]);             // Mish
    y += add_val;                        // + constant
    y = fminf(fmaxf(y, -1.0f), 1.0f);    // Hardtanh clamp
    y *= scale_val;                      // scale
    out[idx] = y;
}

torch::Tensor fused_forward_cuda(torch::Tensor input,
                                 double add_val_d,
                                 double scale_val_d) {
    float add_val   = static_cast<float>(add_val_d);
    float scale_val = static_cast<float>(scale_val_d);

    auto output = torch::empty_like(input);
    const int64_t size = input.numel();

    const int threads = 256;
    const int blocks  = (size + threads - 1) / threads;

    fused_kernel<<<blocks, threads>>>(input.data_ptr<float>(),
                                      output.data_ptr<float>(),
                                      size,
                                      add_val,
                                      scale_val);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_forward_cuda", &fused_forward_cuda,
          "Fused Mish + Add + Hardtanh + Scale (CUDA)");
}
"""

fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["fused_forward_cuda"],
    verbose=False,
)

# ------------------------------------------------------------------
# Optimized model
# ------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Model that performs a transposed convolution and a fused CUDA kernel
    implementing: Mish → add → Hardtanh → scale
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, output_padding, add_value, scale):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride, padding, output_padding
        )
        # Store constants as buffers so they move with .cuda(), .to(device), etc.
        self.register_buffer("add_value_tensor",
                             torch.tensor(float(add_value), dtype=torch.float32))
        self.register_buffer("scale_tensor",
                             torch.tensor(float(scale), dtype=torch.float32))
        self._fused = fused_ops

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x.contiguous()  # guarantee contiguous memory layout
        x = self._fused.fused_forward_cuda(
            x,
            float(self.add_value_tensor.item()),
            float(self.scale_tensor.item()),
        )
        return x

# ------------------------------------------------------------------
# Helpers matching the original API
# ------------------------------------------------------------------
batch_size = 64
in_channels  = 32
out_channels = 32
height = width = 64
kernel_size  = 3
stride       = 2
padding      = 1
output_padding = 1
add_value = 0.5
scale = 2

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, device='cuda')]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride,
            padding, output_padding, add_value, scale]
