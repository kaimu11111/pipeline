import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Inline CUDA/C++ source code implementing a custom softmax along channel dimension (dim=1) for 5D tensors,
# and a custom 3D max-pooling with kernel_size=K (assumed to be stride=K, no padding).
# Note: These are naive reference implementations for demonstration and should compile & run.
source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// -------------------------------------
//  Naive Softmax along dim=1 for 5D tensor
//  shape: [N, C, D, H, W]
// -------------------------------------
__global__ void softmax_3d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int N, const int C, const int D, const int H, const int W)
{
    // Each thread processes exactly one spatial location across channels.
    // We flatten (N, D, H, W) into a single index: idx
    // Then we do a loop over C in that single thread.
    int spatial_size = D * H * W;
    int total_size   = N * spatial_size;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    // Compute n, d, h, w from idx
    int n  = idx / spatial_size;
    int rem = idx % spatial_size;
    int d  = rem / (H * W);
    rem    = rem % (H * W);
    int h  = rem / W;
    int w  = rem % W;

    // 1) find max
    float max_val = -1e30f;
    for (int c = 0; c < C; ++c) {
        int offset = ((n * C + c) * D + d) * H * W + h * W + w;
        float val = input[offset];
        if (val > max_val) max_val = val;
    }

    // 2) compute sum of exps
    float sum_exp = 0.0f;
    for (int c = 0; c < C; ++c) {
        int offset = ((n * C + c) * D + d) * H * W + h * W + w;
        sum_exp += expf(input[offset] - max_val);
    }

    // 3) write out final result
    for (int c = 0; c < C; ++c) {
        int offset = ((n * C + c) * D + d) * H * W + h * W + w;
        output[offset] = expf(input[offset] - max_val) / sum_exp;
    }
}

torch::Tensor softmax3d_cuda(torch::Tensor input) {
    // input shape: [N, C, D, H, W]
    TORCH_CHECK(input.dim() == 5, "Input must have 5 dimensions");
    auto sizes = input.sizes(); 
    int N = sizes[0];
    int C = sizes[1];
    int D = sizes[2];
    int H = sizes[3];
    int W = sizes[4];

    auto output = torch::zeros_like(input);

    int spatial_size = D * H * W;
    int total_size = N * spatial_size; // We'll launch 1 thread for each (N, D, H, W)
    int block_size = 256;
    int grid_size = (total_size + block_size - 1) / block_size;

    softmax_3d_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D, H, W
    );

    return output;
}

// -------------------------------------
//  Naive 3D max-pool with kernel_size=K, stride=K, no padding
//  input  shape: [N, C, D, H, W]
//  output shape: [N, C, D/K, H/K, W/K] (integer division)
// -------------------------------------
__global__ void maxpool3d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int N, const int C,
    const int inD, const int inH, const int inW,
    const int outD, const int outH, const int outW,
    const int K)
{
    // Each thread processes one element in the output
    int out_size = N * C * outD * outH * outW;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_size) return;

    // decode n,c,d,h,w in the output
    int tmp = idx;
    int w_out = tmp % outW; tmp /= outW;
    int h_out = tmp % outH; tmp /= outH;
    int d_out = tmp % outD; tmp /= outD;
    int c_out = tmp % C;    tmp /= C;
    int n_out = tmp;

    // compute where that maps in the input
    int d_in = d_out * K;
    int h_in = h_out * K;
    int w_in = w_out * K;

    float max_val = -1e30f;
    for(int kd = 0; kd < K; kd++){
        for(int kh = 0; kh < K; kh++){
            for(int kw = 0; kw < K; kw++){
                int dd = d_in + kd;
                int hh = h_in + kh;
                int ww = w_in + kw;
                if(dd < inD && hh < inH && ww < inW){
                    int in_offset = (((n_out * C + c_out) * inD + dd) * inH + hh) * inW + ww;
                    float val = input[in_offset];
                    if(val > max_val){
                        max_val = val;
                    }
                }
            }
        }
    }
    output[idx] = max_val;
}

torch::Tensor maxpool3d_cuda(torch::Tensor input, int kernel_size) {
    TORCH_CHECK(input.dim() == 5, "Input must have 5 dimensions");
    auto sizes = input.sizes();
    int N  = sizes[0];
    int C  = sizes[1];
    int D  = sizes[2];
    int H  = sizes[3];
    int W  = sizes[4];

    int outD = D / kernel_size;
    int outH = H / kernel_size;
    int outW = W / kernel_size;

    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto output = torch::empty({N, C, outD, outH, outW}, options);

    int out_size = N * C * outD * outH * outW;
    int block_size = 256;
    int grid_size = (out_size + block_size - 1) / block_size;

    maxpool3d_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D, H, W, outD, outH, outW, kernel_size
    );

    return output;
}
"""

# Declarations for the two custom functions above
cpp_src = r"""
torch::Tensor softmax3d_cuda(torch::Tensor input);
torch::Tensor maxpool3d_cuda(torch::Tensor input, int kernel_size);
"""

# Build the inline extension
custom_ops = load_inline(
    name="custom_3d_ops",
    cpp_sources=cpp_src,
    cuda_sources=source,
    extra_cflags=["-O2"],
    extra_ldflags=[],
    verbose=False,
    functions=["softmax3d_cuda", "maxpool3d_cuda"],
)

class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D convolution, then a custom Softmax, then two custom max pooling operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        x = self.conv(x)
        x = custom_ops.softmax3d_cuda(x)
        x = custom_ops.maxpool3d_cuda(x, self.pool_kernel_size)
        x = custom_ops.maxpool3d_cuda(x, self.pool_kernel_size)
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
pool_kernel_size = 2

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, pool_kernel_size]
