import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_add_ln_kernel(const float* __restrict__ a,
                                    const float* __restrict__ b,
                                    const float* __restrict__ gamma,
                                    const float* __restrict__ beta,
                                    float* __restrict__ out,
                                    int cols,
                                    float eps)
{
    extern __shared__ float sh[];
    const int row    = blockIdx.x;
    const int tid    = threadIdx.x;
    const int stride = blockDim.x;
    const int base   = row * cols;

    /* ---------- mean ---------- */
    float thread_sum = 0.f;
    for (int col = tid; col < cols; col += stride) {
        float v = a[base + col] + b[base + col];
        thread_sum += v;
    }
    sh[tid] = thread_sum;
    __syncthreads();

    for (int s = stride >> 1; s > 0; s >>= 1) {
        if (tid < s)
            sh[tid] += sh[tid + s];
        __syncthreads();
    }
    float mean = sh[0] / static_cast<float>(cols);

    /* ---------- variance ---------- */
    float thread_var = 0.f;
    for (int col = tid; col < cols; col += stride) {
        float v = a[base + col] + b[base + col];
        float d = v - mean;
        thread_var += d * d;
    }
    sh[tid] = thread_var;
    __syncthreads();

    for (int s = stride >> 1; s > 0; s >>= 1) {
        if (tid < s)
            sh[tid] += sh[tid + s];
        __syncthreads();
    }
    float var     = sh[0] / static_cast<float>(cols);
    float inv_std = rsqrtf(var + eps);

    /* ---------- normalize + affine ---------- */
    for (int col = tid; col < cols; col += stride) {
        float v    = a[base + col] + b[base + col];
        float norm = (v - mean) * inv_std;
        out[base + col] = norm * gamma[col] + beta[col];
    }
}

torch::Tensor fused_add_layernorm_cuda(torch::Tensor a,
                                       torch::Tensor b,
                                       torch::Tensor gamma,
                                       torch::Tensor beta,
                                       double eps)
{
    TORCH_CHECK(a.device().is_cuda() && b.device().is_cuda(), "Tensors must be CUDA");
    TORCH_CHECK(a.sizes() == b.sizes(), "Input shapes must match");
    TORCH_CHECK(a.scalar_type() == torch::kFloat32, "Only float32 supported");

    auto out = torch::empty_like(a);

    const int rows = a.size(0) * a.size(1);   // seq_len * batch
    const int cols = a.size(2);

    int tpb = 1;
    while (tpb < cols && tpb < 1024) tpb <<= 1;
    if (tpb > 1024) tpb = 1024;

    dim3 grid(rows);
    dim3 block(tpb);
    size_t shm = tpb * sizeof(float);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    fused_add_ln_kernel<<<grid, block, shm, stream>>>(
        a.data_ptr<float>(), b.data_ptr<float>(),
        gamma.data_ptr<float>(), beta.data_ptr<float>(),
        out.data_ptr<float>(), cols, static_cast<float>(eps));

    return out;
}
"""

cpp_src = r"""
torch::Tensor fused_add_layernorm_cuda(torch::Tensor a,
                                       torch::Tensor b,
                                       torch::Tensor gamma,
                                       torch::Tensor beta,
                                       double eps);
"""

fused_add_ln = load_inline(
    name="fused_add_layernorm",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["fused_add_layernorm_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x_reshaped = x.view(B, C, H * W).permute(2, 0, 1)  # (seq_len, B, C)
        attn_out, _ = self.attn(x_reshaped, x_reshaped, x_reshaped)

        y = fused_add_ln.fused_add_layernorm_cuda(
            attn_out.contiguous(),
            x_reshaped.contiguous(),
            self.norm.weight.contiguous(),
            self.norm.bias.contiguous(),
            self.norm.eps,
        )

        return y.permute(1, 2, 0).view(B, C, H, W)
