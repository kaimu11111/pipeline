import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
from einops import rearrange
import torch.nn.functional as F

source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void segsum_exp_kernel(const float* __restrict__ A,
                                  float* __restrict__ L,
                                  int Llen) {
    /* Each block handles one (b,h,c) triplet. Threads iterate over the
       flattened (i,j) index space of size L². We need the segment-sum of
       A[j+1 … i] so that the diagonal produces exp(0)=1. */
    int bhc = blockIdx.x;                         // which (b,h,c)
    const float* A_ptr = A + bhc * Llen;          // pointer to this sequence
    float* L_ptr       = L + (size_t)bhc * Llen * Llen;

    const int tid         = threadIdx.x;
    const int n_threads   = blockDim.x;
    const int total_elems = Llen * Llen;

    for (int idx = tid; idx < total_elems; idx += n_threads) {
        int i = idx / Llen;          // row
        int j = idx - i * Llen;      // column

        float val = 0.0f;
        if (j <= i) {
            float sum = 0.0f;
            for (int k = j + 1; k <= i; ++k)  // NOTE: start at j+1
                sum += A_ptr[k];
            val = expf(sum);
        }
        L_ptr[idx] = val;             // upper-triangular stays 0
    }
}

torch::Tensor segsum_exp_cuda(torch::Tensor A) {
    TORCH_CHECK(A.dim() == 4, "Expected A of shape (B,H,C,L)");
    TORCH_CHECK(A.is_cuda(), "A must be on CUDA");
    TORCH_CHECK(A.scalar_type() == at::kFloat,
                "Only float32 tensors are supported");

    auto A_contig = A.contiguous();
    const int B = A_contig.size(0);
    const int H = A_contig.size(1);
    const int C = A_contig.size(2);
    const int L = A_contig.size(3);

    auto options = torch::TensorOptions().dtype(A.dtype()).device(A.device());
    auto L_out = torch::zeros({B, H, C, L, L}, options);

    const int grid  = B * H * C;
    const int block = 256;
    segsum_exp_kernel<<<grid, block>>>(
        A_contig.data_ptr<float>(),
        L_out.data_ptr<float>(),
        L);

    return L_out;
}
"""

cpp_src = "torch::Tensor segsum_exp_cuda(torch::Tensor A);"

segsum_exp = load_inline(
    name="segsum_exp",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["segsum_exp_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, batch_size, seq_length, n_heads, d_head, d_state, block_len=64):
        super().__init__()
        assert seq_length % block_len == 0, "Sequence length must be divisible by block length"

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.n_heads    = n_heads
        self.d_head     = d_head
        self.d_state    = d_state
        self.block_len  = block_len

        # Parameters identical to the baseline model
        self.A = nn.Parameter(torch.randn(batch_size, seq_length, n_heads))
        self.B = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        self.C = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))

        self._segsum_exp = segsum_exp.segsum_exp_cuda

    # Reference segment-sum (for tiny tensors / CPU fall-back)
    @staticmethod
    def _segsum_ref(x):
        T = x.size(-1)
        x_cumsum = torch.cumsum(x, dim=-1)
        segsum   = x_cumsum[..., :, None] - x_cumsum[..., None, :]  # (j+1 … i)
        mask     = torch.tril(torch.ones(T, T, device=x.device, dtype=bool))
        segsum   = segsum.masked_fill(~mask, -torch.inf)
        return segsum

    def forward(self, X, initial_states=None):
        # ── Blockify tensors ──────────────────────────────────────────────
        X_blk, A_blk, B_blk, C_blk = [
            rearrange(t, "b (c l) ... -> b c l ...", l=self.block_len)
            for t in (X, self.A, self.B, self.C)
        ]

        # A required as (b,h,c,l) for CUDA kernel
        A_blk_for_k = rearrange(A_blk, "b c l h -> b h c l")

        # ── 1. Diagonal block computation (CUDA) ─────────────────────────
        L = self._segsum_exp(A_blk_for_k)  # (b h c l l)
        Y_diag = torch.einsum(
            "bclhn,bcshn,bhcls,bcshp->bclhp",
            C_blk, B_blk, L, X_blk,
        )

        # ── 2. Intra-chunk state update ──────────────────────────────────
        A_cumsum = torch.cumsum(A_blk_for_k, dim=-1)
        decay_states = torch.exp(A_cumsum[..., :, -1:] - A_cumsum)
        states = torch.einsum(
            "bclhn,bhcl,bclhp->bchpn",
            B_blk, decay_states, X_blk,
        )

        # ── 3. Inter-chunk recurrence ────────────────────────────────────
        if initial_states is None:
            initial_states = torch.zeros_like(states[:, :1])
        states = torch.cat([initial_states, states], dim=1)

        decay_chunk = torch.exp(self._segsum_ref(F.pad(A_cumsum[..., -1], (1, 0))))
        new_states  = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
        states      = new_states[:, :-1]

        # ── 4. State-to-output conversion ────────────────────────────────
        state_decay_out = torch.exp(A_cumsum)
        Y_off = torch.einsum(
            "bclhn,bchpn,bhcl->bclhp",
            C_blk, states, state_decay_out,
        )

        # ── Restore original sequence shape ──────────────────────────────
        Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
        return Y
