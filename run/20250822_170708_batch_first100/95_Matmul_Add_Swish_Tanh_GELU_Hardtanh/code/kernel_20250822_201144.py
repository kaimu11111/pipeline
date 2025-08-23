import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// Fused kernel that performs: 
// 1) out = x * weight^T + bias
// 2) out += add_value
// 3) out = swish(out) = out * sigmoid(out)
// 4) out = tanh(out)
// 5) out = gelu(out)  (approx)
// 6) out = clamp(out, -1, 1)

__device__ __forceinline__ float gelu_approx(float x) {
    // Approximate GELU from:
    //  x * 0.5 * (1 + tanh( sqrt(2/pi) * (x + 0.044715x^3 ) ))
    const float c1 = 0.7978845608f;  // sqrt(2.0 / M_PI)
    const float c2 = 0.044715f;
    float t = c1 * (x + c2 * x * x * x);
    return 0.5f * x * (1.0f + tanhf(t));
}

__global__ void fused_matmul_activation_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ add_value,
    float* __restrict__ out,
    int batch_size,
    int in_features,
    int out_features
) {
    // Linear index for the output matrix of shape [batch_size, out_features]
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = batch_size * out_features;
    if (idx >= total) return;

    // Compute (i, j) from linear index
    int i = idx / out_features;
    int j = idx % out_features;

    // 1) Matmul + bias
    float val = 0.0f;
    for(int k = 0; k < in_features; k++){
        val += x[i * in_features + k] * weight[j * in_features + k];
    }
    val += bias[j];

    // 2) + add_value
    val += add_value[j];

    // 3) swish: val * sigmoid(val)
    float sigmoid_val = 1.0f / (1.0f + expf(-val));
    val = val * sigmoid_val;

    // 4) tanh
    val = tanhf(val);

    // 5) gelu (approx)
    val = gelu_approx(val);

    // 6) clamp to [-1, 1]
    if (val > 1.0f) val = 1.0f;
    if (val < -1.0f) val = -1.0f;

    out[idx] = val;
}

torch::Tensor fused_matmul_activation_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor add_value
) {
    // Shapes:
    int batch_size = x.size(0);
    int in_features = x.size(1);
    int out_features = weight.size(0);

    auto out = torch::empty({batch_size, out_features}, x.options());

    int total = batch_size * out_features;
    const int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;

    fused_matmul_activation_kernel<<<gridSize, blockSize>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        add_value.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );

    return out;
}
'''

cpp_src = r'''
torch::Tensor fused_matmul_activation_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor add_value
);
'''

fused_ops = load_inline(
    name="fused_custom_ops",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["fused_matmul_activation_cuda"],
    verbose=True
)

class ModelNew(nn.Module):
    """
    Optimized model that fuses matmul, bias, add_value, and multiple activations into a single custom CUDA kernel.
    """
    def __init__(self, in_features, out_features, add_value_shape):
        super(ModelNew, self).__init__()
        # Match nn.Linear(in_features, out_features) param shapes
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.add_value = nn.Parameter(torch.randn(add_value_shape))
        self.fused_ops = fused_ops

    def forward(self, x):
        return self.fused_ops.fused_matmul_activation_cuda(x, self.weight, self.bias, self.add_value)
