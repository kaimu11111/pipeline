import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------
# Custom CUDA kernels for matmul+bias+sigmoid and row sum
# ------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// ------------------------------------------------------
// Kernel: matmul + bias + sigmoid
//   x: [batch_size, input_size]
//   w: [hidden_size, input_size]
//   b: [hidden_size]
//   out: [batch_size, hidden_size]
// ------------------------------------------------------
__global__ void matmul_bias_sigmoid_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ out,
    const int batch_size,
    const int input_size,
    const int hidden_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < batch_size * hidden_size) {
        int i = tid / hidden_size;  // row in x
        int j = tid % hidden_size;  // col in w

        float val = 0.0f;
        // naive matmul
        for (int k = 0; k < input_size; k++) {
            val += x[i * input_size + k] * w[j * input_size + k];
        }
        // add bias + sigmoid
        val += b[j];
        val = 1.0f / (1.0f + expf(-val));

        out[tid] = val;
    }
}

// ------------------------------------------------------
// Kernel: row-wise sum
//   in: [batch_size, hidden_size]
//   out: [batch_size, 1] (flattened as [batch_size])
// ------------------------------------------------------
__global__ void row_sum_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    const int batch_size,
    const int hidden_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch_size) {
        float sum_val = 0.0f;
        for (int j = 0; j < hidden_size; j++) {
            sum_val += in[i * hidden_size + j];
        }
        out[i] = sum_val;
    }
}

// ------------------------------------------------------
// Wrapper for matmul+bias+sigmoid
// ------------------------------------------------------
torch::Tensor matmul_bias_sigmoid_cuda(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor b
) {
    auto batch_size = x.size(0);
    auto input_size = x.size(1);
    auto hidden_size = w.size(0);

    // create output [batch_size, hidden_size]
    auto out = torch::empty({batch_size, hidden_size}, x.options());

    int total = batch_size * hidden_size;
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;

    matmul_bias_sigmoid_kernel<<<gridSize, blockSize>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        input_size,
        hidden_size
    );

    return out;
}

// ------------------------------------------------------
// Wrapper for the row-wise sum
// ------------------------------------------------------
torch::Tensor row_sum_cuda(
    torch::Tensor in
) {
    auto batch_size = in.size(0);
    auto hidden_size = in.size(1);

    // output shape: [batch_size, 1] but we'll create a 1D tensor first
    auto out = torch::empty({batch_size}, in.options());

    int blockSize = 256;
    int gridSize = (batch_size + blockSize - 1) / blockSize;

    row_sum_kernel<<<gridSize, blockSize>>>(
        in.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        hidden_size
    );

    // reshape to [batch_size, 1]
    return out.view({batch_size, 1});
}
"""

cpp_declarations = r"""
torch::Tensor matmul_bias_sigmoid_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor b);
torch::Tensor row_sum_cuda(torch::Tensor in);
"""

# Compile the inline CUDA extension
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=cpp_declarations,
    cuda_sources=cuda_source,
    functions=["matmul_bias_sigmoid_cuda", "row_sum_cuda"],
    verbose=False
)

# ------------------------------------------------------
# Optimized Model using the custom CUDA operators
# ------------------------------------------------------
class ModelNew(nn.Module):
    """
    Re-implementation of the model using custom CUDA kernels.
    """
    def __init__(self, input_size, hidden_size):
        super(ModelNew, self).__init__()
        # Replicate linear's weight and bias as parameters
        # We maintain the same shapes as nn.Linear(input_size, hidden_size)
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size) * 0.01)
        self.bias = nn.Parameter(torch.randn(hidden_size) * 0.01)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Output tensor of shape (batch_size, 1).
        """
        # 1) matmul + bias + sigmoid
        out = fused_ops.matmul_bias_sigmoid_cuda(x, self.weight, self.bias)
        # 2) sum across dim=1
        out = fused_ops.row_sum_cuda(out)
        return out
