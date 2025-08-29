import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------------
#                       Custom CUDA kernels (inline build)
# ---------------------------------------------------------------------------

cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ------------------- 2-D row-wise concatenation ----------------------------
__global__ void concat2d_kernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ out,
                                int rows,
                                int cols_a,
                                int cols_b) {
    const int total_cols = cols_a + cols_b;
    const int idx        = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elem = rows * total_cols;

    if (idx >= total_elem) return;

    const int row = idx / total_cols;
    const int col = idx - row * total_cols;

    if (col < cols_a) {
        out[idx] = a[row * cols_a + col];
    } else {
        const int col_b = col - cols_a;
        out[idx] = b[row * cols_b + col_b];
    }
}

torch::Tensor concat2d_cuda(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "Only 2-D tensors are supported");
    TORCH_CHECK(a.size(0) == b.size(0), "Tensors must have the same number of rows");

    const int rows    = a.size(0);
    const int cols_a  = a.size(1);
    const int cols_b  = b.size(1);
    const int out_col = cols_a + cols_b;

    auto out = torch::empty({rows, out_col}, a.options());

    const int total_elem = rows * out_col;
    const int threads    = 256;
    const int blocks     = (total_elem + threads - 1) / threads;

    concat2d_kernel<<<blocks, threads>>>(a.data_ptr<float>(),
                                         b.data_ptr<float>(),
                                         out.data_ptr<float>(),
                                         rows,
                                         cols_a,
                                         cols_b);
    return out;
}

// -------------------- Element-wise tanh activation -------------------------
__global__ void tanh_kernel(const float* __restrict__ in,
                            float* __restrict__       out,
                            int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    out[idx] = tanhf(in[idx]);
}

torch::Tensor tanh_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");

    auto out = torch::empty_like(input);
    const int size    = input.numel();
    const int threads = 256;
    const int blocks  = (size + threads - 1) / threads;

    tanh_kernel<<<blocks, threads>>>(input.data_ptr<float>(),
                                     out.data_ptr<float>(),
                                     size);
    return out;
}
"""

cpp_src = """
torch::Tensor concat2d_cuda(torch::Tensor a, torch::Tensor b);
torch::Tensor tanh_cuda(torch::Tensor input);
"""

# Build & load the CUDA extension
_ops = load_inline(
    name="rnn_fused_ops",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["concat2d_cuda", "tanh_cuda"],
    verbose=False,
)

# ---------------------------------------------------------------------------
#                               Optimised Model
# ---------------------------------------------------------------------------

class ModelNew(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Optimised Vanilla RNN cell using custom CUDA kernels for
        concatenation and tanh activation.
        """
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Register persistent hidden state buffer (not a parameter)
        self.register_buffer("hidden", torch.randn(1, hidden_size))

        # Linear layers
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size, bias=True)
        self.h2o = nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, x: torch.Tensor, initial_hidden: torch.Tensor = None) -> torch.Tensor:
        """
        :param x:  Input tensor (batch, input_size)
        :param initial_hidden: Optional initial hidden state (batch, hidden_size)
        :return:   Output tensor (batch, output_size)
        """
        if initial_hidden is not None:
            # Update hidden buffer in-place to avoid reallocations
            self.hidden = initial_hidden.detach()
        # Ensure hidden is on the correct device and with correct batch size
        if self.hidden.size(0) != x.size(0):
            self.hidden = self.hidden[:1].repeat(x.size(0), 1).to(x.device)
        self.hidden = self.hidden.to(x.device)

        # ------------------------------------------------------------------
        # Fused custom concatenation
        # ------------------------------------------------------------------
        combined = _ops.concat2d_cuda(x.contiguous(), self.hidden.contiguous())

        # Linear projection to hidden
        hidden_lin = self.i2h(combined)

        # ------------------------------------------------------------------
        # Fused custom tanh activation
        # ------------------------------------------------------------------
        self.hidden = _ops.tanh_cuda(hidden_lin)

        # Output layer
        output = self.h2o(self.hidden)
        return output

# ---------------------------------------------------------------------------
#                          Helper functions (unchanged)
# ---------------------------------------------------------------------------

batch_size       = 128
input_size       = 8192
hidden_size      = 8192
output_size      = 4096
sequence_length  = 128

def get_inputs():
    return [torch.rand(batch_size, input_size, device="cuda"),
            torch.rand(batch_size, hidden_size, device="cuda")]

def get_init_inputs():
    return [input_size, hidden_size, output_size]
