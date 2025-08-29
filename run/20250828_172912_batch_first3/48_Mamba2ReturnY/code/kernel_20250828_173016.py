import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
from einops import rearrange

# ---------------------------------------------------------------------------
# CUDA kernel: fused segment-sum + exp with implicit lower-triangular masking
# ---------------------------------------------------------------------------
cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void segsum_exp_kernel(const float* __restrict__ A,
                                  float* __restrict__ L,
                                  int Llen) {
    /*  Each block handles one (b,h,c) triplet.                             *
     *  Threads iterate over the flattened (i,j) index space of size L^2.   *
     *  For invalid (j>i) positions we write 0 (equivalent to exp(-inf)).   */
    int bhc = blockIdx.x;                 // which (b,h,c)
    const float* A_ptr = A + bhc * Llen;  // pointer to start of this row
    float* L_ptr       = L + bhc * Llen * Llen;

    const int tid          = threadIdx.x;
    const int n_threads    = blockDim.x;
    const int total_elems  = Llen * Llen;

    for (int idx = tid; idx < total_elems; idx += n_threads) {
        int i = idx / Llen;   // row
        int j = idx - i * Llen; // column

        float val = 0.f;
        if (j <= i) {
            float sum = 0.f;
            for (int k = j; k <= i; ++k)
                sum += A_ptr[k];
            val = expf(sum);
        }   // else keep val = 0

        L_ptr[idx] = val;
    }
}

torch::Tensor segsum_exp_cuda(torch::Tensor A) {
    TORCH_CHECK(A.dim() == 4, "Expected A of shape (B, H, C, L)");
    TORCH_CHECK(A.is_cuda(),   "A must reside on CUDA device");
    TORCH_CHECK(A.scalar_type() == at::kFloat,
                "Only float32 tensors are supported");

    auto A_contig = A.contiguous();
    const int B = A_contig.size(0);
    const int H = A_contig.size(1);
    const int C = A_contig.size(2);
    const int L = A_contig.size(3);

    auto options = torch::TensorOptions()
                       .dtype(A_contig.dtype())
                       .device(A_contig.device());
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

# Build inline extension
segsum_exp = load_inline(
    name="segsum_exp",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["segsum_exp_cuda"],
    verbose=False,
)

# ---------------------------------------------------------------------------
# Optimised model with custom CUDA kernel
# ---------------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, batch_size, seq_length, n_heads, d_head, d_state, block_len=64):
        super().__init__()
        assert seq_length % block_len == 0, "Sequence length must be divisible by block length"

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        self.block_len = block_len

        # Parameters identical to the baseline model
        self.A = nn.Parameter(torch.randn(batch_size, seq_length, n_heads))
        self.B = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))
        self.C = nn.Parameter(torch.randn(batch_size, seq_length, n_heads, d_state))

        # expose custom kernel
        self._segsum_exp = segsum_exp.segsum_exp_cuda

    # -----------------------------------------------------------------------
    # Naïve segment-sum (kept for smaller tensors that don't use the kernel)
    # -----------------------------------------------------------------------
    @staticmethod
    def _segsum_ref(x):
        T = x.size(-1)
        x_cumsum = torch.cumsum(x, dim=-1)
        x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
        x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
        return x_segsum

    def forward(self, X, initial_states=None):
        # Rearrange into blocks/chunks
        X_blocks, A_blocks, B_blocks, C_blocks = [
            rearrange(x, "b (c l) ... -> b c l ...", l=self.block_len)
            for x in (X, self.A, self.B, self.C)
        ]

        # Prepare A in (b, h, c, l) for kernel
        A_blocks = rearrange(A_blocks, "b c l h -> b h c l")

        # ---------------------------------------------------------------
        # 1. Compute diagonal block outputs using the fused CUDA kernel
        # ---------------------------------------------------------------
        L = self._segsum_exp(A_blocks)  # shape: (b h c l l)
        Y_diag = torch.einsum(
            "bclhn,bcshn,bhcls,bcshp->bclhp",
            C_blocks,
            B_blocks,
            L,
            X_blocks,
        )

        # ---------------------------------------------------------------
        # 2. Compute intra-chunk states (unchanged from baseline)
        # ---------------------------------------------------------------
        A_cumsum = torch.cumsum(A_blocks, dim=-1)
        decay_states = torch.exp((A_cumsum[..., :, -1:] - A_cumsum))
        states = torch.einsum(
            "bclhn,bhcl,bclhp->bchpn",
            B_blocks,
            decay_states,
            X_blocks,
        )

        # ---------------------------------------------------------------
        # 3. Inter-chunk recurrence
        # ---------------------------------------------------------------
        if initial_states is None:
            initial_states = torch.zeros_like(states[:, :1])
        states = torch.cat([initial_states, states], dim=1)

        decay_chunk = torch.exp(
            self._segsum_ref(F.pad(A_cumsum[..., -1], (1, 0)))
        )  # retained reference implementation for this small tensor
        new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
        states = new_states[:, :-1]

        # ---------------------------------------------------------------
        # 4. State-to-output conversion
        # ---------------------------------------------------------------
        state_decay_out = torch.exp(A_cumsum)
        Y_off = torch.einsum(
            "bclhn,bchpn,bhcl->bclhp",
            C_blocks,
            states,
            state_decay_out,
        )

        # Combine diagonal/off-diagonal terms and restore sequence shape
        Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
        return Y
