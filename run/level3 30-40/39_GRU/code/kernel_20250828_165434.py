import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import math

# ---------------------------------------------------------------------------
# CUDA kernel + host implementation
# ---------------------------------------------------------------------------
source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float sigmoidf(float x){
    return 1.f / (1.f + expf(-x));
}

__global__ void gru_cell_kernel(const float* __restrict__ gates_x,   // (B,3H)
                                const float* __restrict__ gates_h,   // (B,3H)
                                const float* __restrict__ h_prev,    // (B,H)
                                float* __restrict__ h_new,           // (B,H)
                                int hidden_size, int batch_size){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int H   = hidden_size;
    const int N   = batch_size * H;
    if(idx >= N) return;

    const int b = idx / H;              // batch index
    const int h = idx - b * H;          // hidden index

    const int base   = b * 3 * H + h;   // offset for r gate
    const int base_z = base + H;        // offset for z gate
    const int base_n = base + 2*H;      // offset for n gate

    // retrieve gate pre-activations
    const float x_r = gates_x[base];
    const float x_z = gates_x[base_z];
    const float x_n = gates_x[base_n];

    const float h_r = gates_h[base];
    const float h_z = gates_h[base_z];
    const float h_n = gates_h[base_n];

    // element-wise GRU equations
    const float r   = sigmoidf(x_r + h_r);
    const float z   = sigmoidf(x_z + h_z);
    const float n   = tanhf(x_n + r * h_n);

    const float prev = h_prev[idx];
    const float out  = (1.f - z) * n + z * prev;

    h_new[idx] = out;
}

// Python-visible wrapper
torch::Tensor gru_cell_forward(torch::Tensor gates_x,
                               torch::Tensor gates_h,
                               torch::Tensor h_prev){
    TORCH_CHECK(gates_x.is_cuda(), "gates_x must be a CUDA tensor");
    TORCH_CHECK(gates_h.is_cuda(), "gates_h must be a CUDA tensor");
    TORCH_CHECK(h_prev.is_cuda(),  "h_prev must be a CUDA tensor");

    const int batch_size  = h_prev.size(0);
    const int hidden_size = h_prev.size(1);

    auto h_new = torch::empty_like(h_prev);

    const int total   = batch_size * hidden_size;
    const int threads = 256;
    const int blocks  = (total + threads - 1) / threads;

    gru_cell_kernel<<<blocks, threads>>>(gates_x.data_ptr<float>(),
                                         gates_h.data_ptr<float>(),
                                         h_prev.data_ptr<float>(),
                                         h_new.data_ptr<float>(),
                                         hidden_size,
                                         batch_size);

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }

    return h_new;
}
"""

# ---------------------------------------------------------------------------
# C++ prototypes required by the auto-generated binding file
# ---------------------------------------------------------------------------
cpp_src = r"""
#include <torch/extension.h>

torch::Tensor gru_cell_forward(torch::Tensor gates_x,
                               torch::Tensor gates_h,
                               torch::Tensor h_prev);
"""

# ---------------------------------------------------------------------------
# Build / load the extension
# ---------------------------------------------------------------------------
fused_gru_cell = load_inline(
    name="fused_gru_cell",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["gru_cell_forward"],
    verbose=False,
)

# ---------------------------------------------------------------------------
# PyTorch module using the fused kernel
# ---------------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 bias=True,
                 batch_first=False):
        super().__init__()
        self.input_size   = input_size
        self.hidden_size  = hidden_size
        self.num_layers   = num_layers
        self.bias         = bias
        self.batch_first  = batch_first

        self._weight_ih = []
        self._weight_hh = []
        self._bias_ih   = []
        self._bias_hh   = []

        for l in range(num_layers):
            in_feat = input_size if l == 0 else hidden_size

            w_ih = nn.Parameter(torch.empty(3 * hidden_size, in_feat))
            w_hh = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
            self.register_parameter(f"weight_ih_l{l}", w_ih)
            self.register_parameter(f"weight_hh_l{l}", w_hh)
            self._weight_ih.append(w_ih)
            self._weight_hh.append(w_hh)

            if bias:
                b_ih = nn.Parameter(torch.empty(3 * hidden_size))
                b_hh = nn.Parameter(torch.empty(3 * hidden_size))
                self.register_parameter(f"bias_ih_l{l}", b_ih)
                self.register_parameter(f"bias_hh_l{l}", b_hh)
                self._bias_ih.append(b_ih)
                self._bias_hh.append(b_hh)
            else:
                self.register_parameter(f"bias_ih_l{l}", None)
                self.register_parameter(f"bias_hh_l{l}", None)
                self._bias_ih.append(None)
                self._bias_hh.append(None)

        self.reset_parameters()

        # expose the CUDA kernel
        self.gru_cell_cuda = fused_gru_cell.gru_cell_forward

    def reset_parameters(self):
        # Mimic nn.GRU’s initialization so that weights match reference model
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for param in self.parameters():
            if param is not None:
                nn.init.uniform_(param, -stdv, stdv)

    def _gru_layer(self, x, h, layer_idx):
        """
        x : (seq_len, batch, input_size or hidden_size)
        h : (batch, hidden_size)
        """
        W_ih = self._weight_ih[layer_idx]
        W_hh = self._weight_hh[layer_idx]
        b_ih = self._bias_ih[layer_idx] if self.bias else None
        b_hh = self._bias_hh[layer_idx] if self.bias else None

        seq_len, _, _ = x.shape
        outputs = []

        for t in range(seq_len):
            x_t     = x[t]                                            # (B, F)
            gates_x = torch.nn.functional.linear(x_t, W_ih, b_ih)     # (B, 3H)
            gates_h = torch.nn.functional.linear(h,   W_hh, b_hh)     # (B, 3H)

            h = self.gru_cell_cuda(
                gates_x.contiguous(),
                gates_h.contiguous(),
                h.contiguous()
            )
            outputs.append(h.unsqueeze(0))

        return torch.cat(outputs, 0), h

    def forward(self, x, h_0):
        """
        x   : (seq_len, batch, input_size)  if batch_first = False
        h_0 : (num_layers, batch, hidden_size)
        """
        device = self._weight_ih[0].device
        x   = x.to(device)
        h_0 = h_0.to(device)

        if self.batch_first:
            x = x.transpose(0, 1)  # (seq_len, batch, feat)

        h_n_all   = []
        layer_inp = x

        for l in range(self.num_layers):
            h_l0             = h_0[l]                        # (B, H)
            layer_out, h_lN  = self._gru_layer(layer_inp, h_l0, l)
            h_n_all.append(h_lN.unsqueeze(0))
            layer_inp        = layer_out                     # feed to next layer

        output = layer_out                                   # (seq_len, batch, H)
        h_n    = torch.cat(h_n_all, 0)                       # (num_layers, batch, H)

        if self.batch_first:
            output = output.transpose(0, 1)                  # (batch, seq_len, H)

        return output, h_n
