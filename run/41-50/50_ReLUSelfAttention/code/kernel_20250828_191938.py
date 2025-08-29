import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------------
# CUDA kernel : fused causal-scaled-dot-product + ReLU attention
# ---------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void fused_relu_attention_kernel(
    const float  *__restrict__ q,
    const float  *__restrict__ k,
    const float  *__restrict__ v,
    float        *__restrict__ y,
    const int B,
    const int H,
    const int T,
    const int D,
    const float scale)
{
    /*  Each block:
            blockIdx.x -------->  (B * H)
            blockIdx.y -------->  target token index i            (0 .. T-1)
        Threads (threadIdx.x) ----> head-dim index d              (0 .. D-1)
        The kernel computes y[b,h,i,d]  =  Σ_{j<=i} ReLU((q·k)/√D) * v[j,d]
        Causal mask is enforced by the upper-triangular loop limit (j<=i).
    */

    const int bh = blockIdx.x;
    const int i  = blockIdx.y;
    const int d  = threadIdx.x;
    if (d >= D) return;

    const int b = bh / H;
    const int h = bh % H;

    const size_t base      = ((size_t)b * H + h) * T * D;
    const float *q_ptr     = q + (base + (size_t)i * D);
    const float *k_base    = k + base;
    const float *v_base    = v + base;
          float *y_ptr     = y + (base + (size_t)i * D);

    float acc = 0.0f;
    __shared__ float dot_shared;     // broadcasted (q·k) for current (i,j)

    for (int j = 0; j <= i; ++j)       // causal loop
    {
        if (threadIdx.x == 0) {
            const float *k_ptr = k_base + (size_t)j * D;
            float dot = 0.0f;
            #pragma unroll
            for (int dd = 0; dd < D; ++dd)
                dot += q_ptr[dd] * k_ptr[dd];
            dot_shared = dot * scale;    // scaled
        }
        __syncthreads();
        const float dot_val = dot_shared;   // broadcast

        if (dot_val > 0.0f) {
            const float *v_ptr = v_base + (size_t)j * D;
            acc += dot_val * v_ptr[d];      // ReLU(dot) == max(dot,0)
        }
        __syncthreads();
    }
    y_ptr[d] = acc;
}

torch::Tensor fused_relu_attention(torch::Tensor q,
                                   torch::Tensor k,
                                   torch::Tensor v)
{
    TORCH_CHECK(q.is_cuda(), "q must be CUDA");
    const int B = q.size(0);
    const int H = q.size(1);
    const int T = q.size(2);
    const int D = q.size(3);

    const float scale = 1.0f / sqrtf((float)D);

    auto y = torch::empty_like(q);

    dim3 grid(B * H, T);
    int  block = D <= 1024 ? D : 1024;

    fused_relu_attention_kernel<<<grid, block>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        y.data_ptr<float>(),
        B, H, T, D, scale);

    return y;
}
"""

cpp_decls = "torch::Tensor fused_relu_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v);"

attention_cuda = load_inline(
    name="fused_relu_attention",
    cpp_sources=cpp_decls,
    cuda_sources=cuda_source,
    functions=["fused_relu_attention"],
    verbose=False,
)

# ---------------------------------------------------------------------------
# Optimised model with fused CUDA attention
# ---------------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, n_embd, n_head, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.c_attn = nn.Linear(n_embd, 3 * n_embd)   # q,k,v together

        # causal mask kept only for shape compatibility (not used by CUDA kernel)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(max_seqlen, max_seqlen))
                 .view(1, 1, max_seqlen, max_seqlen)
        )

    def forward(self, x):
        B, T, C = x.size()          # (batch, sequence, emb)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2).contiguous()
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2).contiguous()
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2).contiguous()

        # fused scaled-dot-product + ReLU attention
        y = attention_cuda.fused_relu_attention(q, k, v)

        # (B, T, C) : merge heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return y

# ---------------------------------------------------------------------------
# Helper functions required by the evaluation harness
# ---------------------------------------------------------------------------
batch_size = 8
max_seqlen = 512
n_embd = 768
n_head = 12

def get_inputs():
    return [torch.rand(batch_size, max_seqlen, n_embd).cuda()]

def get_init_inputs():
    return [n_embd, n_head, max_seqlen]
