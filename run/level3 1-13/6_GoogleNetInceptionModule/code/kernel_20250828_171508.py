import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -------------------------------------------------------------------------
# Custom CUDA kernel: channel-wise concatenation for four NCHW tensors
# -------------------------------------------------------------------------
cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void channel_concat_kernel(
    const float* __restrict__ in1,
    const float* __restrict__ in2,
    const float* __restrict__ in3,
    const float* __restrict__ in4,
    float* __restrict__ out,
    int N, int C1, int C2, int C3, int C4,
    int H, int W)
{
    int spatial     = H * W;
    int outC        = C1 + C2 + C3 + C4;
    long total_elem = (long)N * outC * spatial;

    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elem)
        return;

    int n  = idx / (outC * spatial);
    int rem = idx % (outC * spatial);
    int c  = rem / spatial;
    int s  = rem % spatial;   // index within H*W

    const float* src_ptr;
    int src_c, src_channels;

    if (c < C1) {                       // from in1
        src_ptr      = in1;
        src_c        = c;
        src_channels = C1;
    } else if (c < C1 + C2) {           // from in2
        src_ptr      = in2;
        src_c        = c - C1;
        src_channels = C2;
    } else if (c < C1 + C2 + C3) {      // from in3
        src_ptr      = in3;
        src_c        = c - C1 - C2;
        src_channels = C3;
    } else {                            // from in4
        src_ptr      = in4;
        src_c        = c - C1 - C2 - C3;
        src_channels = C4;
    }

    long src_idx = ((long)n * src_channels + src_c) * spatial + s;
    out[idx] = src_ptr[src_idx];
}

torch::Tensor channel_concat_cuda(torch::Tensor in1,
                                  torch::Tensor in2,
                                  torch::Tensor in3,
                                  torch::Tensor in4)
{
    TORCH_CHECK(in1.is_cuda() && in2.is_cuda() && in3.is_cuda() && in4.is_cuda(),
                "All inputs must reside on CUDA");

    TORCH_CHECK(in1.dtype() == torch::kFloat32 &&
                in2.dtype() == torch::kFloat32 &&
                in3.dtype() == torch::kFloat32 &&
                in4.dtype() == torch::kFloat32,
                "Only float32 tensors are supported");

    int N  = in1.size(0);
    int C1 = in1.size(1);
    int H  = in1.size(2);
    int W  = in1.size(3);

    TORCH_CHECK(in2.size(0) == N && in3.size(0) == N && in4.size(0) == N, "Batch size mismatch");
    TORCH_CHECK(in2.size(2) == H && in3.size(2) == H && in4.size(2) == H, "Height mismatch");
    TORCH_CHECK(in2.size(3) == W && in3.size(3) == W && in4.size(3) == W, "Width mismatch");

    int C2 = in2.size(1);
    int C3 = in3.size(1);
    int C4 = in4.size(1);

    int outC = C1 + C2 + C3 + C4;

    auto out = torch::empty({N, outC, H, W}, in1.options());

    long total_elem  = (long)N * outC * H * W;
    int  threads     = 256;
    int  blocks      = (total_elem + threads - 1) / threads;

    channel_concat_kernel<<<blocks, threads>>>(
        in1.data_ptr<float>(), in2.data_ptr<float>(),
        in3.data_ptr<float>(), in4.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C1, C2, C3, C4, H, W);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));

    return out;
}
"""

cpp_src = r"""
torch::Tensor channel_concat_cuda(torch::Tensor in1,
                                  torch::Tensor in2,
                                  torch::Tensor in3,
                                  torch::Tensor in4);
"""

channel_concat = load_inline(
    name="channel_concat",
    cpp_sources=[cpp_src],
    cuda_sources=[cuda_src],
    functions=["channel_concat_cuda"],
    verbose=False,
)

# -------------------------------------------------------------------------
# Optimised model utilising the custom CUDA concat kernel
# -------------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3,
                 reduce_5x5, out_5x5, pool_proj):
        super().__init__()

        # 1×1 conv branch
        self.branch1x1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)

        # 3×3 conv branch
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_3x3, kernel_size=1),
            nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1),
        )

        # 5×5 conv branch
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_5x5, kernel_size=1),
            nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2),
        )

        # Pooling branch
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
        )

    def forward(self, x):
        b1 = self.branch1x1(x)
        b2 = self.branch3x3(x)
        b3 = self.branch5x5(x)
        b4 = self.branch_pool(x)

        # Use the high-performance CUDA concat
        out = channel_concat.channel_concat_cuda(
            b1.contiguous(), b2.contiguous(),
            b3.contiguous(), b4.contiguous()
        )
        return out


# -------------------------------------------------------------------------
# Helper functions (unchanged interface)
# -------------------------------------------------------------------------
def get_inputs(batch_size=2, in_channels=480, height=112, width=112):
    return [torch.rand(batch_size, in_channels, height, width, device="cuda")]

def get_init_inputs():
    # (in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5,
    #  out_5x5, pool_proj)
    return [480, 192, 96, 208, 16, 48, 64]
