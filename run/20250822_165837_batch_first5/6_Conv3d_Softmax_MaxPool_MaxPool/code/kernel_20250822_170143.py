import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Inline CUDA/C++ source for custom softmax and maxpool3d
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// ---------------------------------------------
// Naive softmax over channel dimension (dim=1) for a 5D tensor
// Shape of input: (N, C, D, H, W)
// ---------------------------------------------
__global__ void softmax_3d_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  int N, int C, int D, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int outer_size = N * D * H * W;  // number of "rows" we compute along channel dim

    if (idx < outer_size) {
        // Decode n, d, h, w from idx
        int n = idx / (D * H * W);
        int r = idx % (D * H * W);
        int d = r / (H * W);
        r = r % (H * W);
        int h = r / W;
        int w = r % W;

        // Find max in channel dimension (for numerical stability)
        float max_val = -3.402823e+38f;  // float min
        for (int c = 0; c < C; c++) {
            float val = input[n*C*D*H*W + c*D*H*W + d*H*W + h*W + w];
            max_val = fmaxf(val, max_val);
        }

        // Compute sum of exp(...)
        float sum_exp = 0.f;
        for (int c = 0; c < C; c++) {
            float val = input[n*C*D*H*W + c*D*H*W + d*H*W + h*W + w];
            sum_exp += expf(val - max_val);
        }

        // Write normalized result
        for (int c = 0; c < C; c++) {
            float val = input[n*C*D*H*W + c*D*H*W + d*H*W + h*W + w];
            float softmax_val = expf(val - max_val) / sum_exp;
            output[n*C*D*H*W + c*D*H*W + d*H*W + h*W + w] = softmax_val;
        }
    }
}

// Wrapper for softmax_3d_kernel
torch::Tensor softmax_3d_cuda(torch::Tensor input) {
    TORCH_CHECK(input.dim() == 5, "Input must be 5D (N, C, D, H, W)");
    int N = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);

    auto output = torch::empty_like(input);
    int outer_size = N * D * H * W;

    const int threads = 256;
    const int blocks = (outer_size + threads - 1) / threads;
    softmax_3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D, H, W
    );

    return output;
}

// ---------------------------------------------
// Naive 3D max pooling with kernel size (kD, kH, kW) and stride = (kD, kH, kW)
// Shape of input: (N, C, D, H, W)
// Shape of output: (N, C, D/kD, H/kH, W/kW)
// ---------------------------------------------
__global__ void maxpool3d_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 int N, int C, int D, int H, int W,
                                 int kD, int kH, int kW) {
    // Compute output dimensions
    int D_out = D / kD;
    int H_out = H / kH;
    int W_out = W / kW;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out = N * C * D_out * H_out * W_out;

    if (idx < total_out) {
        // Decode n, c, d_out, h_out, w_out from idx
        int n = idx / (C * D_out * H_out * W_out);
        int r = idx % (C * D_out * H_out * W_out);
        int c = r / (D_out * H_out * W_out);
        r = r % (D_out * H_out * W_out);
        int d_out = r / (H_out * W_out);
        r = r % (H_out * W_out);
        int h_out = r / W_out;
        int w_out = r % W_out;

        // Compute the start indices in the input
        int d_start = d_out * kD;
        int h_start = h_out * kH;
        int w_start = w_out * kW;

        float max_val = -3.402823e+38f;  // float min
        for(int kd = 0; kd < kD; kd++){
            for(int kh = 0; kh < kH; kh++){
                for(int kw_ = 0; kw_ < kW; kw_++){
                    int d_in = d_start + kd;
                    int h_in = h_start + kh;
                    int w_in = w_start + kw_;
                    float val = input[n*C*D*H*W + c*D*H*W + d_in*H*W + h_in*W + w_in];
                    max_val = fmaxf(val, max_val);
                }
            }
        }
        output[idx] = max_val;
    }
}

// Wrapper for maxpool3d_kernel
torch::Tensor maxpool3d_cuda(torch::Tensor input, int kD, int kH, int kW) {
    TORCH_CHECK(input.dim() == 5, "Input must be 5D (N, C, D, H, W)");
    int N = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);

    TORCH_CHECK(D % kD == 0, "D must be divisible by kD");
    TORCH_CHECK(H % kH == 0, "H must be divisible by kH");
    TORCH_CHECK(W % kW == 0, "W must be divisible by kW");

    int D_out = D / kD;
    int H_out = H / kH;
    int W_out = W / kW;

    auto output = torch::empty({N, C, D_out, H_out, W_out}, input.options());
    int total_out = N * C * D_out * H_out * W_out;

    const int threads = 256;
    const int blocks = (total_out + threads - 1) / threads;
    maxpool3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D, H, W,
        kD, kH, kW
    );

    return output;
}

""";

cpp_src = r"""
torch::Tensor softmax_3d_cuda(torch::Tensor input);
torch::Tensor maxpool3d_cuda(torch::Tensor input, int kD, int kH, int kW);
""";

# Build the custom CUDA extension
cudaop = load_inline(
    name="fusion_3d_ops",
    cpp_sources=cpp_src,
    cuda_sources=cuda_source,
    functions=["softmax_3d_cuda", "maxpool3d_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model that uses custom CUDA operators for softmax and maxpool3d,
    while preserving PyTorch's nn.Conv3d for the 3D convolution step.
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        x = self.conv(x)
        x = cudaop.softmax_3d_cuda(x)
        x = cudaop.maxpool3d_cuda(x, self.pool_kernel_size, self.pool_kernel_size, self.pool_kernel_size)
        x = cudaop.maxpool3d_cuda(x, self.pool_kernel_size, self.pool_kernel_size, self.pool_kernel_size)
        return x


batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
pool_kernel_size = 2

def get_inputs():
    """
    Return a list of input tensors with the correct shape on GPU.
    """
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]

def get_init_inputs():
    """
    Return the model initialization parameters.
    """
    return [in_channels, out_channels, kernel_size, pool_kernel_size]
