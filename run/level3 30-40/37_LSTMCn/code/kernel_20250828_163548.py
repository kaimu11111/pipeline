import math
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------------
# CUDA kernel + C++ interface
# ------------------------------------------------------------------
cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void last_timestep_linear_kernel(const scalar_t* __restrict__ lstm_out,
                                            const int seq_len,
                                            const scalar_t* __restrict__ weight,
                                            const scalar_t* __restrict__ bias,
                                            scalar_t* __restrict__ out,
                                            const int hidden_size,
                                            const int output_size) {
    /*  Grid layout
        blockIdx.x -> batch index
        blockIdx.y -> output feature index
        Every block reduces `hidden_size` products to one scalar.
    */
    const int batch_idx = blockIdx.x;
    const int out_idx   = blockIdx.y;

    extern __shared__ char smem[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(smem);

    const int tid = threadIdx.x;

    // Pointer to the last time-step hidden state for this batch
    const scalar_t* last_hidden = lstm_out +
                                  ((batch_idx * seq_len + (seq_len - 1)) * hidden_size);

    // Pointer to the row of the weight matrix that corresponds to this output feature
    const scalar_t* weight_row = weight + out_idx * hidden_size;

    scalar_t accum = static_cast<scalar_t>(0);

    // Strided loop over hidden dimension
    for (int h = tid; h < hidden_size; h += blockDim.x) {
        accum += last_hidden[h] * weight_row[h];
    }

    sdata[tid] = accum;
    __syncthreads();

    // Parallel block-wide reduction
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Write result with bias
    if (tid == 0) {
        out[batch_idx * output_size + out_idx] = sdata[0] + bias[out_idx];
    }
}

torch::Tensor last_timestep_linear(torch::Tensor lstm_out,
                                   torch::Tensor weight,
                                   torch::Tensor bias) {
    TORCH_CHECK(lstm_out.is_cuda(), "lstm_out must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(),  "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(),    "bias must be a CUDA tensor");
    TORCH_CHECK(lstm_out.is_contiguous(), "lstm_out must be contiguous");

    const int batch_size   = lstm_out.size(0);
    const int seq_len      = lstm_out.size(1);
    const int hidden_size  = lstm_out.size(2);
    const int output_size  = bias.size(0);

    auto out = torch::empty({batch_size, output_size}, lstm_out.options());

    const int threads = 256;
    dim3 blocks(batch_size, output_size);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(lstm_out.scalar_type(),
                                        "last_timestep_linear_kernel", ([&] {
        last_timestep_linear_kernel<scalar_t>
            <<<blocks, threads, threads * sizeof(scalar_t)>>>(
                lstm_out.data_ptr<scalar_t>(),
                seq_len,
                weight.data_ptr<scalar_t>(),
                bias.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                hidden_size,
                output_size);
    }));

    return out;
}
"""

cpp_src = "torch::Tensor last_timestep_linear(torch::Tensor lstm_out, torch::Tensor weight, torch::Tensor bias);"

fused_linear = load_inline(
    name="fused_last_timestep_linear",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["last_timestep_linear"],
    extra_cflags=["-O3"],
    verbose=False,
)

# ------------------------------------------------------------------
# Optimised model using the fused CUDA kernel
# ------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(ModelNew, self).__init__()
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=False)

        # Replace nn.Linear by explicit Parameters that will be consumed by the fused kernel
        weight = torch.empty(output_size, hidden_size)
        bias   = torch.empty(output_size)
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
        bound = 1.0 / math.sqrt(fan_in)
        nn.init.uniform_(bias, -bound, bound)

        self.fc_weight = nn.Parameter(weight)
        self.fc_bias   = nn.Parameter(bias)

    def forward(self, x, h0, c0):
        # LSTM forward
        out, state = self.lstm(x, (h0, c0))        # out: (B, T, H)

        # Fused "select last time-step + Linear"
        out = out.contiguous()                     # ensure contiguous memory
        _ = fused_linear.last_timestep_linear(out,
                                              self.fc_weight,
                                              self.fc_bias)
        # The original model discards `out` and returns c_n
        return state[1]
