import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Inline CUDA/C++ source
source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

// -------------------------------------
// HardSwish Kernel
// out[i] = x[i] * ReLU6(x[i] + 3) / 6
// -------------------------------------
__global__ void apply_hardswish_kernel(const float* __restrict__ inp,
                                       float* __restrict__ out,
                                       int64_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = inp[idx];
        float relu6_val = fminf(fmaxf(val + 3.0f, 0.0f), 6.0f);
        out[idx] = val * (relu6_val / 6.0f);
    }
}

torch::Tensor apply_hardswish_cuda(torch::Tensor inp) {
    auto out = torch::empty_like(inp);
    int64_t size = inp.numel();

    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    apply_hardswish_kernel<<<blocks, threads>>>(
        inp.data_ptr<float>(),
        out.data_ptr<float>(),
        size
    );
    return out;
}

// ---------------------------------------------------------------------
// GroupNorm + forward pass in one function for demonstration.
//
// Steps:
// 1) HardSwish already done externally
// 2) Compute per-group mean/var across (C_in_group, D, H, W) for each
//    (batch, group).
// 3) Apply normalization using gamma, beta and store in out.
//
// We do it in two CUDA kernels internally, but expose a single function.
// ---------------------------------------------------------------------

// Kernel 1: Compute sums and squared sums for each (b, g).
__global__ void groupnorm_sum_kernel(
    const float* __restrict__ inp,
    float* __restrict__ sum,
    float* __restrict__ sum_sq,
    int B, int C, int D, int H, int W,
    int groups, int channels_per_group)
{
    // Each thread will handle one element of B*C*D*H*W
    // We'll accumulate sum/group wise via atomic operations
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * D * H * W;
    if (idx < total) {
        int w_off = idx % W;
        int hw = idx / W;
        int h_off = hw % H;
        int dh = hw / H;
        int d_off = dh % D;
        int cd = dh / D;
        int c_off = cd % C;
        int b_off = cd / C;

        int g = c_off / channels_per_group; // group index

        float val = inp[idx];
        atomicAdd(&sum[b_off*groups + g], val);
        atomicAdd(&sum_sq[b_off*groups + g], val * val);
    }
}

// Kernel 2: Apply normalization using computed mean/var for each group.
__global__ void groupnorm_apply_kernel(
    const float* __restrict__ inp,
    float* __restrict__ out,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float eps,
    int B, int C, int D, int H, int W,
    int groups, int channels_per_group)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * D * H * W;
    if (idx < total) {
        int w_off = idx % W;
        int hw = idx / W;
        int h_off = hw % H;
        int dh = hw / H;
        int d_off = dh % D;
        int cd = dh / D;
        int c_off = cd % C;
        int b_off = cd / C;

        int g = c_off / channels_per_group;
        float m = mean[b_off*groups + g];
        float v = var[b_off*groups + g];
        float scale = gamma[c_off] / sqrtf(v + eps);
        float shift = beta[c_off] - scale * m;

        float val = inp[idx];
        out[idx] = val * scale + shift;
    }
}

torch::Tensor group_norm_forward_cuda(
    torch::Tensor inp,
    torch::Tensor gamma,
    torch::Tensor beta,
    int groups,
    float eps)
{
    // inp is shape [B, C, D, H, W]
    int B = inp.size(0);
    int C = inp.size(1);
    int D = inp.size(2);
    int H = inp.size(3);
    int W = inp.size(4);

    // channels per group
    int channels_per_group = C / groups;

    auto out = torch::empty_like(inp);

    // 1) Compute sum and sum_sq.
    auto sums = torch::zeros({B, groups}, inp.options());
    auto sums_sq = torch::zeros({B, groups}, inp.options());

    int64_t total_elems = B * C * D * H * W;
    const int threads = 256;
    const int blocks = (total_elems + threads - 1) / threads;

    groupnorm_sum_kernel<<<blocks, threads>>>(
        inp.data_ptr<float>(),
        sums.data_ptr<float>(),
        sums_sq.data_ptr<float>(),
        B, C, D, H, W, groups, channels_per_group
    );

    // 2) Compute mean & var on CPU (or GPU). We'll do it on CPU for simplicity.
    //    Transfer sums, sums_sq back if needed. Then transfer mean/var again.
    //    For demonstration, we do it on GPU to avoid D->H->D copies.

    // We'll do it in-place on sums, sums_sq on GPU in a small kernel launch:
    // Actually let's do it with a simple pointwise approach on the GPU:
    // We'll store mean in sums, var in sums_sq for convenience. Then pass them
    // to groupnorm_apply_kernel.

    // We'll do a small device kernel to compute sums => mean, sums_sq => var
    // for each b, g.
    const int stats_size = B * groups;

    __global__ void compute_mean_var_kernel(
        float* sum, float* sum_sq,
        int channels_per_group, int D, int H, int W)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int area = channels_per_group * D * H * W;
        if (idx < %stats_size%) {
            float m = sum[idx] / (float)area;
            float v = sum_sq[idx] / (float)area - m * m;
            sum[idx] = m;
            sum_sq[idx] = v;
        }
    }

    // Ugly hack: we generate code as a string because we need to embed stats_size:
    // We'll do it manually below.

    return out; // This won't compile as is. We'll fix it properly next.
}
"""

# To keep things fully functional in a single block, let's define it all again
# but inline the method to compute mean/var with a separate approach. 
# We'll finalize a working version below:

fixed_source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// -------------------------------------
// HardSwish Kernel
// -------------------------------------
__global__ void apply_hardswish_kernel(const float* inp,
                                       float* out,
                                       int64_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = inp[idx];
        float relu6_val = fminf(fmaxf(val + 3.0f, 0.0f), 6.0f);
        out[idx] = val * (relu6_val / 6.0f);
    }
}

torch::Tensor apply_hardswish_cuda(torch::Tensor inp) {
    auto out = torch::empty_like(inp);
    int64_t size = inp.numel();

    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    apply_hardswish_kernel<<<blocks, threads>>>(
        inp.data_ptr<float>(),
        out.data_ptr<float>(),
        size
    );
    return out;
}

// ------------------------------------------------------
// groupnorm_sum_kernel: compute sum & sum of squares
// for each (batch, group), across channels_in_group*D*H*W
// ------------------------------------------------------
__global__ void groupnorm_sum_kernel(
    const float* inp,
    float* sum,
    float* sum_sq,
    int B, int C, int D, int H, int W,
    int groups, int channels_per_group)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * D * H * W;
    if (idx < total) {
        int w_off = idx % W;
        int hw = idx / W;
        int h_off = hw % H;
        int dh = hw / H;
        int d_off = dh % D;
        int cd = dh / D;
        int c_off = cd % C;
        int b_off = cd / C;

        int g = c_off / channels_per_group; 
        float val = inp[idx];
        atomicAdd(&sum[b_off*groups + g], val);
        atomicAdd(&sum_sq[b_off*groups + g], val * val);
    }
}

// ----------------------------------
// compute_mean_var_kernel:
// sum => mean, sum_sq => var
// ----------------------------------
__global__ void compute_mean_var_kernel(
    float* sum,
    float* sum_sq,
    int B,
    int groups,
    int channels_per_group,
    int D,
    int H,
    int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * groups;
    float denom = (float)(channels_per_group * D * H * W);
    if (idx < total) {
        float m = sum[idx] / denom;
        float v = sum_sq[idx] / denom - m*m;
        sum[idx] = m;        // store mean
        sum_sq[idx] = v;     // store var
    }
}

// ----------------------------------
// groupnorm_apply_kernel:
// out = (inp - mean) / sqrt(var+eps)*gamma + beta
// ----------------------------------
__global__ void groupnorm_apply_kernel(
    const float* inp,
    float* out,
    const float* mean,
    const float* var,
    const float* gamma,
    const float* beta,
    float eps,
    int B, int C, int D, int H, int W,
    int groups, int channels_per_group)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * D * H * W;
    if (idx < total) {
        int w_off = idx % W;
        int hw = idx / W;
        int h_off = hw % H;
        int dh = hw / H;
        int d_off = dh % D;
        int cd = dh / D;
        int c_off = cd % C;
        int b_off = cd / C;

        int g = c_off / channels_per_group;
        float m = mean[b_off*groups + g];
        float v = var[b_off*groups + g];
        float scale = gamma[c_off] / sqrtf(v + eps);
        float shift = beta[c_off] - scale * m;

        float val = inp[idx];
        out[idx] = val * scale + shift;
    }
}

// Orchestrator function
torch::Tensor group_norm_forward_cuda(
    torch::Tensor inp,
    torch::Tensor gamma,
    torch::Tensor beta,
    int groups,
    float eps)
{
    // [B, C, D, H, W]
    int B = inp.size(0);
    int C = inp.size(1);
    int D = inp.size(2);
    int H = inp.size(3);
    int W = inp.size(4);

    int channels_per_group = C / groups;
    auto out = torch::empty_like(inp);

    auto sums = torch::zeros({B, groups}, inp.options());
    auto sums_sq = torch::zeros({B, groups}, inp.options());

    int64_t total_elems = (int64_t)B*C*D*H*W;
    {
      const int threads = 256;
      const int blocks = (total_elems + threads - 1) / threads;
      groupnorm_sum_kernel<<<blocks, threads>>>(
         inp.data_ptr<float>(),
         sums.data_ptr<float>(),
         sums_sq.data_ptr<float>(),
         B, C, D, H, W,
         groups, channels_per_group
      );
    }

    // compute mean/var in sums/sums_sq
    {
      const int threads = 256;
      const int blocks = (B*groups + threads - 1) / threads;
      compute_mean_var_kernel<<<blocks, threads>>>(
         sums.data_ptr<float>(),
         sums_sq.data_ptr<float>(),
         B,
         groups,
         channels_per_group,
         D,
         H,
         W
      );
    }

    // apply group norm
    {
      const int threads = 256;
      const int blocks = (total_elems + threads - 1) / threads;
      groupnorm_apply_kernel<<<blocks, threads>>>(
         inp.data_ptr<float>(),
         out.data_ptr<float>(),
         sums.data_ptr<float>(),    // means
         sums_sq.data_ptr<float>(), // vars
         gamma.data_ptr<float>(),
         beta.data_ptr<float>(),
         eps,
         B, C, D, H, W,
         groups, channels_per_group
      );
    }

    return out;
}

// ----------------------------------
// Mean pool across dims [2,3,4] => shape (B, C)
// That is mean over D*H*W
// ----------------------------------
__global__ void mean_pool_3d_kernel(
    const float* inp,
    float* out,
    int B, int C,
    int D, int H, int W)
{
    // each (b, c) is a single element in out
    // total threads = B*C
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B*C;
    if (idx < total) {
        int c_off = idx % C;
        int b_off = idx / C;

        float sum_val = 0.0f;
        int area = D*H*W;
        int base = (b_off*C + c_off)*D*H*W;
        for(int i = 0; i < area; i++){
            sum_val += inp[base + i];
        }
        out[idx] = sum_val / (float)area;
    }
}

torch::Tensor mean_pool_3d_cuda(torch::Tensor inp) {
    // inp: [B, C, D, H, W]
    int B = inp.size(0);
    int C = inp.size(1);
    int D = inp.size(2);
    int H = inp.size(3);
    int W = inp.size(4);

    auto out = torch::empty({B, C}, inp.options());

    int total = B*C;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    mean_pool_3d_kernel<<<blocks, threads>>>(
        inp.data_ptr<float>(),
        out.data_ptr<float>(),
        B, C,
        D, H, W
    );
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("apply_hardswish_cuda", &apply_hardswish_cuda, "Apply HardSwish (CUDA)");
    m.def("group_norm_forward_cuda", &group_norm_forward_cuda, "GroupNorm forward (CUDA)");
    m.def("mean_pool_3d_cuda", &mean_pool_3d_cuda, "Mean pool over dims [2,3,4] (CUDA)");
}
'''

cpp_src = r"""
torch::Tensor apply_hardswish_cuda(torch::Tensor inp);
torch::Tensor group_norm_forward_cuda(torch::Tensor inp,
                                      torch::Tensor gamma,
                                      torch::Tensor beta,
                                      int groups,
                                      float eps);
torch::Tensor mean_pool_3d_cuda(torch::Tensor inp);
"""

# Build the custom CUDA extension
custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=[cpp_src],
    cuda_sources=[fixed_source],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    functions=[
        "apply_hardswish_cuda",
        "group_norm_forward_cuda",
        "mean_pool_3d_cuda"
    ],
    verbose=False
)

# Optimized model
class ModelNew(nn.Module):
    """
    Model that performs:
    1. Conv3D (pytorch)
    2. HardSwish activation (custom CUDA)
    3. GroupNorm (custom CUDA)
    4. Mean pooling across spatial dimensions (custom CUDA)
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)
        # We'll replicate group norm parameters:
        self.gamma = nn.Parameter(torch.ones(out_channels))
        self.beta = nn.Parameter(torch.zeros(out_channels))
        self.num_groups = num_groups

    def forward(self, x):
        # 1) standard conv
        x = self.conv(x)
        # 2) custom HardSwish
        x = custom_ops.apply_hardswish_cuda(x)
        # 3) custom GroupNorm
        x = custom_ops.group_norm_forward_cuda(x, self.gamma, self.beta, self.num_groups, 1e-5)
        # 4) mean pool over D,H,W
        x = custom_ops.mean_pool_3d_cuda(x)
        return x

# Same input helper functions
def get_inputs():
    batch_size = 1024
    in_channels = 3
    depth, height, width = 16, 32, 32
    return [torch.rand(batch_size, in_channels, depth, height, width).cuda()]

def get_init_inputs():
    in_channels = 3
    out_channels = 16
    kernel_size = 4
    return [in_channels, out_channels, kernel_size]
