import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------------
# CUDA source
# ---------------------------------------------------------------------------
source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be float32")

// Kernel: one thread per (flattened) sample
__global__ void segsum_exp_kernel(const float* __restrict__ x,
                                  float* __restrict__ out,
                                  const int D,  // flattened batch-head-chunk count
                                  const int T)  // sequence length per sample
{
    const int d = blockIdx.x;
    if (d >= D) return;

    const float* x_ptr = x   + d * T;
    float*       o_ptr = out + d * T * T;

    for (int i = 0; i < T; ++i) {
        float acc = 0.0f;

        // Lower triangle (j <= i) — EXCLUSIVE prefix, so diagonal becomes 1
        for (int j = i; j >= 0; --j) {
            o_ptr[i * T + j] = __expf(acc);  // store before adding x[j]
            acc += x_ptr[j];
        }

        // Upper triangle (j > i)
        for (int j = i + 1; j < T; ++j) {
            o_ptr[i * T + j] = -INFINITY;
        }
    }
}

torch::Tensor segsum_exp_cuda(torch::Tensor x) {
    CHECK_CUDA(x);
    CHECK_CONTIGUOUS(x);
    CHECK_FLOAT(x);

    const int64_t T   = x.size(-1);
    const int64_t D64 = x.numel() / T;
    TORCH_CHECK(D64 <= INT_MAX, "Tensor too large");
    const int D = static_cast<int>(D64);

    // Output shape: x.shape + [T]
    std::vector<int64_t> out_sizes = x.sizes().vec();
    out_sizes.push_back(T);
    auto out = torch::empty(out_sizes, x.options());

    segsum_exp_kernel<<<D, 1>>>(x.data_ptr<float>(),
                                out.data_ptr<float>(),
                                D, static_cast<int>(T));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(err));
    return out;
}
"""

# ---------------------------------------------------------------------------
# C++ prototypes
# ---------------------------------------------------------------------------
cpp_src = r"""
torch::Tensor segsum_exp_cuda(torch::Tensor x);
"""

# ---------------------------------------------------------------------------
# Build / load
# ---------------------------------------------------------------------------
segsum_exp = load_inline(
    name="segsum_exp_fixed",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["segsum_exp_cuda"],
    verbose=False,
)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self,
                 batch_size,
                 seq_length,
                 n_heads,
                 d_head,
                 d_state,
                 block_len=64):
        super().__init__()
        assert seq_length % block_len == 0, "Sequence length must be divisible by block length"

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.block_len = block_len

        # Parameters
        self.A = nn.Parameter(torch.randn(batch_size, seq_length, n_heads))
        self.B = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        self.C = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))

        self.segsum_exp = segsum_exp

    def forward(self, X, initial_states=None):
        # Step 1: block-wise rearrangement
        X_blk, A_blk, B_blk, C_blk = [
            torch.einsum("b (c l) ... -> b c l ...", x, self.block_len)
            if False else x.reshape(self.batch_size, self.seq_length // self.block_len, self.block_len, *x.shape[2:])
            for x in (X, self.A, self.B, self.C)
        ]

        # Step 2: A-related terms
        A_blk = A_blk.permute(0, 3, 1, 2).contiguous()        # [b, h, c, l]
        A_cumsum = torch.cumsum(A_blk, dim=-1)

        L = self.segsum_exp.segsum_exp_cuda(A_blk)            # [b, h, c, l, l]

        # Step 3: diagonal block outputs
        C_e = C_blk                                           # [b, c, l, h, n]
        B_e = B_blk                                           # [b, c, l, h, n]
        X_e = X_blk                                           # [b, c, l, h, p]

        Y_diag = torch.einsum("bclhn,bclhn,bhcll,bclhp->bclhp",
                              C_e, B_e, L, X_e)

        # Step 4: intra-chunk states
        decay_states = torch.exp(A_cumsum[..., -1:] - A_cumsum)
        states = torch.einsum("bclhn,bhcl,bclhp->bchpn",
                              B_blk, decay_states, X_blk)

        # Step 5: inter-chunk recurrence
        if initial_states is None:
            initial_states = torch.zeros_like(states[:, :1])

        states = torch.cat([initial_states, states], dim=1)

        last_acum = A_cumsum[..., -1]                          # [b, h, c]
        last_acum_pad = torch.nn.functional.pad(last_acum, (1, 0))
        decay_chunk = self.segsum_exp.segsum_exp_cuda(last_acum_pad)

        new_states = torch.einsum("bhzc,bchpn->bzhpn",
                                  decay_chunk, states)
        return new_states[:, -1]
