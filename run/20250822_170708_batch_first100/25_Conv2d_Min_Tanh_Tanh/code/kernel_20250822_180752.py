import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void min_and_double_tanh_kernel(const float* __restrict__ input,
                                           float* __restrict__ output,
                                           int N, int C, int H, int W) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * H * W;
    if (i < total) {
        int n = i / (H * W);
        int hw = i % (H * W);
        int h = hw / W;
        int w = hw % W;

        // Find the minimum value across the channel dimension
        float min_val = input[n * C * H * W + 0 * H * W + h * W + w];
        for (int c = 1; c < C; c++) {
            float val = input[n * C * H * W + c * H * W + h * W + w];
            if (val < min_val) {
                min_val = val;
            }
        }

        // Apply tanh twice
        float t = tanhf(min_val);
        float out_val = tanhf(t);

        // Write the result (shape: [N, 1, H, W])
        output[n * H * W + h * W + w] = out_val;
    }
}

torch::Tensor min_and_double_tanh_cuda(torch::Tensor input) {
    auto N = input.size(0);
    auto C = input.size(1);
    auto H = input.size(2);
    auto W = input.size(3);

    auto output = torch::zeros({N, 1, H, W}, input.options());

    int total = N * H * W;
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;

    min_and_double_tanh_kernel<<<gridSize, blockSize>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W
    );
    return output;
}
"""

cpp_src = r"""
torch::Tensor min_and_double_tanh_cuda(torch::Tensor input);
"""

min_and_double_tanh = load_inline(
    name="min_and_double_tanh",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["min_and_double_tanh_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a convolution, then a custom fused min + double Tanh kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.min_and_double_tanh = min_and_double_tanh

    def forward(self, x):
        x = self.conv(x)
        x = self.min_and_double_tanh.min_and_double_tanh_cuda(x)
        return x


batch_size = 128
in_channels = 16
out_channels = 64
height = width = 256
kernel_size = 3

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
