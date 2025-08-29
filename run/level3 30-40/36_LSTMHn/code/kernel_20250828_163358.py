import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------------
# Custom CUDA kernel: takes the LAST time-step of an LSTM output and performs
# a linear projection (weight * x + bias) in a single fused kernel.
# ---------------------------------------------------------------------------
cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void last_timestep_linear_kernel(const scalar_t* __restrict__ x,      // (B, S, H)
                                            const scalar_t* __restrict__ weight, // (O, H)
                                            const scalar_t* __restrict__ bias,   // (O)
                                            scalar_t* __restrict__ out,          // (B, O)
                                            int B, int S, int H, int O) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;   // output neuron
    int b = blockIdx.y;                              // batch index
    if (o >= O || b >= B) return;

    // pointer to the last timestep of sample b
    const scalar_t* x_ptr = x + ((b * S + (S - 1)) * H);
    const scalar_t* w_ptr = weight + o * H;

    scalar_t acc = bias[o];
    for (int h = 0; h < H; ++h)
        acc += w_ptr[h] * x_ptr[h];

    out[b * O + o] = acc;
}

torch::Tensor last_timestep_linear_cuda(torch::Tensor x,
                                        torch::Tensor weight,
                                        torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda() && weight.is_cuda() && bias.is_cuda(),
                "All tensors must be CUDA tensors");
    TORCH_CHECK(x.dtype() == torch::kFloat && weight.dtype() == torch::kFloat &&
                bias.dtype() == torch::kFloat,
                "Only float32 tensors are supported");

    const int B = x.size(0);
    const int S = x.size(1);
    const int H = x.size(2);
    const int O = weight.size(0);

    auto out = torch::empty({B, O}, x.options());

    const int threads = 256;
    const dim3 grid((O + threads - 1) / threads, B);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "last_timestep_linear_cuda", ([&] {
        last_timestep_linear_kernel<scalar_t><<<grid, threads>>>(
            x.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            B, S, H, O);
    }));

    return out;
}
"""

cpp_src = """
torch::Tensor last_timestep_linear_cuda(torch::Tensor x,
                                        torch::Tensor weight,
                                        torch::Tensor bias);
"""

last_ts_linear = load_inline(
    name="last_ts_linear",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["last_timestep_linear_cuda"],
    verbose=False,
)

# ---------------------------------------------------------------------------
# Optimised model
# ---------------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self._last_ts_linear = last_ts_linear

    def forward(self, x, h0, c0):
        # x : (B, S, I)
        # h0, c0 : (num_layers, B, H)
        lstm_out, state = self.lstm(x, (h0, c0))  # lstm_out: (B, S, H)

        # -------------------------------------------------------------------
        # Original model executed a linear layer on the LAST timestep but
        # ignored the result. We reproduce this behaviour with the fused kernel
        # for timing-equivalent semantics, then return state[0].
        # -------------------------------------------------------------------
        _ = self._last_ts_linear.last_timestep_linear_cuda(
            lstm_out.contiguous(),
            self.fc.weight.contiguous(),
            self.fc.bias.contiguous(),
        )

        return state[0]

# ---------------------------------------------------------------------------
# Helpers to create random inputs, mirroring the original scaffold
# ---------------------------------------------------------------------------
batch_size = 5
sequence_length = 128
input_size = 64
hidden_size = 128
num_layers = 6
output_size = 10
dropout = 0.0

def get_inputs():
    return [
        torch.rand(batch_size, sequence_length, input_size, device="cuda"),
        torch.rand((num_layers, batch_size, hidden_size), device="cuda"),
        torch.rand((num_layers, batch_size, hidden_size), device="cuda"),
    ]

def get_init_inputs():
    return [input_size, hidden_size, num_layers, output_size, dropout]
