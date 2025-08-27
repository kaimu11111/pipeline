import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------------
# Custom CUDA kernel: column-wise scaling for a 2-D tensor
# ------------------------------------------------------------------
cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void scale_mul_kernel(const scalar_t* __restrict__ x,
                                 const scalar_t* __restrict__ scale,
                                 scalar_t* __restrict__ out,
                                 int rows,
                                 int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx < total) {
        int col = idx % cols;           // feature dimension
        out[idx] = x[idx] * scale[col]; // element-wise multiply
    }
}

torch::Tensor scale_mul_cuda(torch::Tensor x, torch::Tensor scale) {
    x = x.contiguous();
    scale = scale.contiguous();

    const int rows = x.size(0);
    const int cols = x.size(1);
    const int total = rows * cols;

    auto out = torch::empty_like(x);

    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "scale_mul_cuda", ([&] {
        scale_mul_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            scale.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            rows,
            cols);
    }));

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scale_mul_cuda", &scale_mul_cuda, "Column-wise scale multiply (CUDA)");
}
"""

cpp_decls = "torch::Tensor scale_mul_cuda(torch::Tensor x, torch::Tensor scale);"

scale_mul = load_inline(
    name="scale_mul",
    cpp_sources=cpp_decls,
    cuda_sources=cuda_src,
    functions=["scale_mul_cuda"],
    verbose=False,
)

# ------------------------------------------------------------------
# Optimized model using the custom kernel
# ------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Optimized model:
      1. GEMM via nn.Linear
      2. Custom CUDA kernel for column-wise scaling
      3. BatchNorm1d
    """
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)
        self.scale_mul = scale_mul

    def forward(self, x):
        x = self.gemm(x)
        x = self.scale_mul.scale_mul_cuda(x, self.scale)
        x = self.bn(x)
        return x

# ------------------------------------------------------------------
# Helpers for external use
# ------------------------------------------------------------------
batch_size = 512
in_features = 4096
out_features = 4096
scale_shape = (out_features,)

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, scale_shape]
