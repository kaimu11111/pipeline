import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------
# Custom CUDA kernels: Fast GELU + Causal (masked) Softmax
# -----------------------------------------------------------------------
cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

////////////////////////////////////////////////////////////////
// Fast GELU kernel
////////////////////////////////////////////////////////////////
__global__ void fast_gelu_kernel(const float* __restrict__ x,
                                 float* __restrict__ out,
                                 int64_t numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    float val = x[idx];
    float inner = 0.7978845608028654f * (val + 0.044715f * val * val * val); // sqrt(2/pi)=0.79788456
    out[idx] = 0.5f * val * (1.0f + tanhf(inner));
}

torch::Tensor fast_gelu_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "input must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat, "only float tensors supported");

    auto out = torch::empty_like(x);
    int64_t numel = x.numel();

    const int threads = 256;
    const int blocks  = (numel + threads - 1) / threads;
    fast_gelu_kernel<<<blocks, threads>>>(x.data_ptr<float>(),
                                          out.data_ptr<float>(),
                                          numel);
    return out;
}

////////////////////////////////////////////////////////////////
// Causal (masked) Softmax kernel
// Each thread processes one sequence position j in a row (i)
// Shared-memory reduction for max and sum to improve numerical stability
////////////////////////////////////////////////////////////////
template<int BLOCK_SIZE>
__global__ void causal_softmax_kernel(const float* __restrict__ input,
                                      float* __restrict__ output,
                                      int BxH,          // B * nHeads
                                      int T) {          // sequence length
    // Each block works on a single (batch, head, i) row
    int row = blockIdx.x;
    if (row >= BxH * T) return;

    int i   = row % T;           // current timestep (for masking)
    int seq = row / T;           // (B * nHeads) index

    const float* row_in  = input  + row * T;
    float*       row_out = output + row * T;

    // Compute max (masked)
    __shared__ float s_max;
    float thread_max = -FLT_MAX;
    for (int j = threadIdx.x; j <= i; j += BLOCK_SIZE) {
        thread_max = fmaxf(thread_max, row_in[j]);
    }

    // reduction to find global max
    if (threadIdx.x == 0) s_max = -FLT_MAX;
    __syncthreads();

    atomicMax((int*)&s_max, __float_as_int(thread_max));
    __syncthreads();
    float max_val = s_max;

    // Compute exp and sum of exp
    __shared__ float s_sum;
    float thread_sum = 0.0f;
    for (int j = threadIdx.x; j <= i; j += BLOCK_SIZE) {
        float val = __expf(row_in[j] - max_val);
        row_out[j] = val;  // store temporarily
        thread_sum += val;
    }

    if (threadIdx.x == 0) s_sum = 0.0f;
    __syncthreads();
    atomicAdd(&s_sum, thread_sum);
    __syncthreads();
    float sum_val = s_sum;
    float inv_sum = 1.0f / sum_val;

    // Normalize & write results; zero for j>i (masked positions)
    for (int j = threadIdx.x; j < T; j += BLOCK_SIZE) {
        if (j <= i) {
            row_out[j] *= inv_sum;
        } else {
            row_out[j] = 0.0f;   // masked
        }
    }
}

torch::Tensor causal_softmax_cuda(torch::Tensor att) {
    TORCH_CHECK(att.is_cuda(), "input must be CUDA");
    TORCH_CHECK(att.scalar_type() == torch::kFloat, "only float tensors supported");
    TORCH_CHECK(att.dim() == 4, "att tensor must be (B, nH, T, T)");

    auto out = torch::empty_like(att);
    int B     = att.size(0);
    int nH    = att.size(1);
    int T     = att.size(2);    // att shape: (B, nH, T, T)
    int rows  = B * nH * T;     // total rows

    const int BLOCK_SIZE = 128; // tune if necessary
    dim3 block(BLOCK_SIZE);
    dim3 grid(rows);

    causal_softmax_kernel<BLOCK_SIZE><<<grid, block>>>(
        att.data_ptr<float>(),
        out.data_ptr<float>(),
        B * nH,
        T);

    return out;
}

////////////////////////////////////////////////////////////////
// PyBind
////////////////////////////////////////////////////////////////
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fast_gelu_cuda", &fast_gelu_cuda, "Fast GELU forward (CUDA)");
    m.def("causal_softmax_cuda", &causal_softmax_cuda, "Causal masked softmax (CUDA)");
}
"""

cpp_decls = """
torch::Tensor fast_gelu_cuda(torch::Tensor x);
torch::Tensor causal_softmax_cuda(torch::Tensor att);
"""

kernels = load_inline(
    name="custom_fused_ops",
    cpp_sources=cpp_decls,
    cuda_sources=cuda_src,
    functions=["fast_gelu_cuda", "causal_softmax_cuda"],
    verbose=False,
)

# -----------------------------------------------------------------------
# Python modules wrapping the kernels
# -----------------------------------------------------------------------
class FastGELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return kernels.fast_gelu_cuda(x.contiguous())

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention using custom CUDA causal softmax.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=True)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=True)

        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)

        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = kernels.causal_softmax_cuda(att.contiguous())
        att = self.attn_dropout(att)

        y = torch.matmul(att, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

# -----------------------------------------------------------------------
# Optimized Transformer Block
# -----------------------------------------------------------------------
class ModelNew(nn.Module):
    """Optimized Transformer block with custom CUDA kernels"""

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(n_embd, 4 * n_embd, bias=True),
            c_proj  = nn.Linear(4 * n_embd, n_embd, bias=True),
            act     = FastGELU(),
            dropout = nn.Dropout(resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x
