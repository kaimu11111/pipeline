# 1. Imports ──────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# 2. CUDA source ──────────────────────────────────────────────────────
source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math_constants.h>

// Inclusive segment-sum with log-domain upper-triangular masking.
// For each sequence A[0..L-1] (per (B,H,C) triple) produce
//   S_{i,j} = Σ_{k=j}^{i} A_k     (0 ≤ j ≤ i < L)
// Upper-triangular (j>i) is set to -∞ so that exp(S) becomes 0.
__global__ void segsum_exp_kernel(const float* __restrict__ A,
                                  float*       __restrict__ L,
                                  const int                Llen)
{
    const int bhc       = blockIdx.x;                        // packed (B,H,C)
    const float* A_ptr  = A + (size_t)bhc * Llen;            // sequence start
    float*       L_ptr  = L + (size_t)bhc * Llen * Llen;     // out-matrix start

    const int tid         = threadIdx.x;
    const int n_threads   = blockDim.x;
    const int total_elems = Llen * Llen;

    for (int idx = tid; idx < total_elems; idx += n_threads)
    {
        const int i = idx / Llen;        // row  (target position)
        const int j = idx - i * Llen;    // col  (source position)

        float log_sum;
        if (j <= i)                      // lower-triangular incl. diagonal
        {
            float acc = 0.0f;
            #pragma unroll
            for (int k = j; k <= i; ++k)   // Σ_{k=j}^{i}  (inclusive)
                acc += A_ptr[k];
            log_sum = acc;                 // log-weight
        }
        else
        {
            log_sum = -CUDART_INF_F;       // mask with −∞  (exp⇒0)
        }
        L_ptr[idx] = __expf(log_sum);      // write exp(sum)
    }
}

torch::Tensor segsum_exp_cuda(torch::Tensor A)
{
    TORCH_CHECK(A.dim() == 4, "Expected A of shape (B,H,C,L)");
    TORCH_CHECK(A.is_cuda(), "Tensor must reside on CUDA device");
    TORCH_CHECK(A.scalar_type() == at::kFloat, "Only float32 supported");

    auto A_contig = A.contiguous();
    const int B    = A_contig.size(0);
    const int H    = A_contig.size(1);
    const int C    = A_contig.size(2);
    const int Llen = A_contig.size(3);

    auto opts = torch::TensorOptions().dtype(A.dtype()).device(A.device());
    auto out  = torch::empty({B, H, C, Llen, Llen}, opts);

    const int grid  = B * H * C;
    const int block = 256;

    segsum_exp_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        A_contig.data_ptr<float>(),
        out.data_ptr<float>(),
        Llen);

    return out;
}
"""

# 3. C++ prototypes ───────────────────────────────────────────────────
cpp_src = "torch::Tensor segsum_exp_cuda(torch::Tensor A);"

# 4. load_inline call ─────────────────────────────────────────────────
segsum_exp = load_inline(
    name         = "segsum_exp",
    cpp_sources  = cpp_src,
    cuda_sources = source,
    functions    = ["segsum_exp_cuda"],
    verbose      = False,
)

# 5. Python wrapper module ────────────────────────────────────────────
class ModelNew(nn.Module):
    def __init__(self, batch_size, seq_length, n_heads, d_head, d_state, block_len=64):
        super().__init__()
        assert seq_length % block_len == 0, "seq_len must be divisible by block_len"

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.n_heads    = n_heads
        self.d_head     = d_head
        self.d_state    = d_state
        self.block_len  = block_len

        # parameters copied from baseline
        self.A = nn.Parameter(torch.randn(batch_size, seq_length, n_heads))
        self.B = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        self.C = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))

        self._segsum_exp = segsum_exp.segsum_exp_cuda

    # Inclusive CPU implementation (matches CUDA kernel)
    @staticmethod
    def _segsum_ref(x: torch.Tensor) -> torch.Tensor:
        T      = x.size(-1)
        cumsum = torch.cumsum(x, dim=-1)
        segsum = cumsum[..., :, None] - cumsum[..., None, :] + x[..., None, :]
        mask   = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        segsum = segsum.masked_fill(~mask, float("-inf"))
        return segsum

    # reshape helpers
    def _blockify(self, t: torch.Tensor) -> torch.Tensor:
        b, s = t.shape[:2]
        c    = s // self.block_len
        return t.view(b, c, self.block_len, *t.shape[2:])

    def forward(self, X, initial_states=None):
        # ── blockify ──────────────────────────────────────────────
        X_blk = self._blockify(X)
        A_blk = self._blockify(self.A)
        B_blk = self._blockify(self.B)
        C_blk = self._blockify(self.C)

        # (b,h,c,l) for kernel
        A_blk_k = A_blk.permute(0, 3, 1, 2)                    # (B,H,C,L)

        # ── 1. diagonal block via CUDA ───────────────────────────
        L     = self._segsum_exp(A_blk_k)                      # (B,H,C,L,L)
        Y_dig = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp",
                             C_blk, B_blk, L, X_blk)

        # ── 2. intra-chunk state update ─────────────────────────
        A_cumsum    = torch.cumsum(A_blk_k, dim=-1)
        decay_state = torch.exp(A_cumsum[..., :, -1:] - A_cumsum)
        states      = torch.einsum("bclhn,bhcl,bclhp->bchpn",
                                   B_blk, decay_state, X_blk)

        # ── 3. inter-chunk recurrence ───────────────────────────
        if initial_states is None:
            initial_states = torch.zeros_like(states[:, :1])
        states = torch.cat([initial_states, states], dim=1)

        decay_chunk = torch.exp(self._segsum_ref(
            nn.functional.pad(A_cumsum[..., -1], (1, 0))
        ))
        new_states  = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
        states      = new_states[:, :-1]

        # ── 4. state-to-output conversion ───────────────────────
        state_decay = torch.exp(A_cumsum)
        Y_off       = torch.einsum("bclhn,bchpn,bhcl->bclhp",
                                   C_blk, states, state_decay)

        # ── restore original shape ──────────────────────────────
        Y_tot = Y_dig + Y_off
        b, c, l, h, p = Y_tot.shape
        return Y_tot.reshape(b, c * l, h, p)
