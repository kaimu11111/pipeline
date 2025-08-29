import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------------
# CUDA kernel + C++ wrapper
# ---------------------------------------------------------------------------
source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__device__ __forceinline__ scalar_t sigmoidf(scalar_t x) {
    return scalar_t(1) / (scalar_t(1) + expf(-x));
}

template <typename scalar_t>
__global__ void gru_step_kernel(
        const scalar_t* __restrict__ x_r,
        const scalar_t* __restrict__ x_z,
        const scalar_t* __restrict__ x_n,
        const scalar_t* __restrict__ h_r,
        const scalar_t* __restrict__ h_z,
        const scalar_t* __restrict__ h_n,
        const scalar_t* __restrict__ h_prev,
        scalar_t*       __restrict__ h_out,
        int hidden_size,
        int total_elems)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elems) return;

    scalar_t r  = sigmoidf(x_r[idx] + h_r[idx]);
    scalar_t z  = sigmoidf(x_z[idx] + h_z[idx]);
    scalar_t n  = tanhf(x_n[idx] + r * h_n[idx]);
    scalar_t hp = h_prev[idx];
    h_out[idx]  = (scalar_t(1) - z) * n + z * hp;
}

std::vector<torch::Tensor> gru_step_cuda(
        torch::Tensor x_r,
        torch::Tensor x_z,
        torch::Tensor x_n,
        torch::Tensor h_r,
        torch::Tensor h_z,
        torch::Tensor h_n,
        torch::Tensor h_prev) {

    TORCH_CHECK(x_r.is_cuda(), "All tensors must be on CUDA");
    int batch_size  = x_r.size(0);
    int hidden_size = x_r.size(1);
    int total       = batch_size * hidden_size;

    auto h_out = torch::empty_like(h_prev);

    const int threads = 256;
    const int blocks  = (total + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x_r.scalar_type(), "gru_step_cuda", ([&] {
        gru_step_kernel<scalar_t><<<blocks, threads>>>(
            x_r.template data_ptr<scalar_t>(),
            x_z.template data_ptr<scalar_t>(),
            x_n.template data_ptr<scalar_t>(),
            h_r.template data_ptr<scalar_t>(),
            h_z.template data_ptr<scalar_t>(),
            h_n.template data_ptr<scalar_t>(),
            h_prev.template data_ptr<scalar_t>(),
            h_out.template data_ptr<scalar_t>(),
            hidden_size,
            total);
    }));
    return {h_out};
}
"""

cpp_src = r"""
std::vector<torch::Tensor> gru_step_cuda(
        torch::Tensor x_r,
        torch::Tensor x_z,
        torch::Tensor x_n,
        torch::Tensor h_r,
        torch::Tensor h_z,
        torch::Tensor h_n,
        torch::Tensor h_prev);
"""

gru_cuda = load_inline(
    name         = "gru_fused_step",
    cpp_sources  = cpp_src,
    cuda_sources = source,
    functions    = ["gru_step_cuda"],
    verbose      = False,
)

# ---------------------------------------------------------------------------
# PyTorch wrapper
# ---------------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Multi-layer GRU (unidirectional) whose per-time-step update is executed by
    a custom fused CUDA kernel.  Parameters are registered with the **exact**
    names used by nn.GRU, enabling state_dict interoperability.
    """
    def __init__(self, input_size, hidden_size, num_layers=1,
                 bias=True, batch_first=False):
        super().__init__()
        self.input_size   = input_size
        self.hidden_size  = hidden_size
        self.num_layers   = num_layers
        self.batch_first  = batch_first
        self.bias         = bias

        stdv = 1.0 / hidden_size ** 0.5
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            # weight_ih_l{k}
            w_ih = nn.Parameter(torch.empty(3 * hidden_size, layer_input_size))
            nn.init.uniform_(w_ih, -stdv, stdv)
            self.register_parameter(f"weight_ih_l{layer}", w_ih)

            # weight_hh_l{k}
            w_hh = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
            nn.init.uniform_(w_hh, -stdv, stdv)
            self.register_parameter(f"weight_hh_l{layer}", w_hh)

            if bias:
                b_ih = nn.Parameter(torch.empty(3 * hidden_size))
                b_hh = nn.Parameter(torch.empty(3 * hidden_size))
                nn.init.uniform_(b_ih, -stdv, stdv)
                nn.init.uniform_(b_hh, -stdv, stdv)
                self.register_parameter(f"bias_ih_l{layer}", b_ih)
                self.register_parameter(f"bias_hh_l{layer}", b_hh)

    def _layer_step(self, layer, x_t, h_prev):
        w_ih = getattr(self, f"weight_ih_l{layer}")
        w_hh = getattr(self, f"weight_hh_l{layer}")
        b_ih = getattr(self, f"bias_ih_l{layer}") if self.bias else None
        b_hh = getattr(self, f"bias_hh_l{layer}") if self.bias else None

        gi = torch.nn.functional.linear(x_t, w_ih, b_ih)   # (B, 3*H)
        gh = torch.nn.functional.linear(h_prev, w_hh, b_hh)

        gi_r, gi_z, gi_n = gi.chunk(3, dim=1)
        gh_r, gh_z, gh_n = gh.chunk(3, dim=1)

        h_out = gru_cuda.gru_step_cuda(
            gi_r.contiguous(), gi_z.contiguous(), gi_n.contiguous(),
            gh_r.contiguous(), gh_z.contiguous(), gh_n.contiguous(),
            h_prev.contiguous()
        )[0]
        return h_out

    def forward(self, x, h0):
        # x : (seq, batch, feat)  |  (batch, seq, feat) if batch_first
        if self.batch_first:
            x = x.transpose(0, 1)               # -> (seq, batch, feat)

        seq_len, batch_size, _ = x.size()
        h_t = list(torch.unbind(h0, dim=0))     # len == num_layers

        for t in range(seq_len):
            inp = x[t]
            for l in range(self.num_layers):
                h_new = self._layer_step(l, inp, h_t[l])
                inp   = h_new
                h_t[l] = h_new

        return torch.stack(h_t, dim=0)          # (num_layers, batch, hidden)
