import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


# ---------------------------------------------------------------------------
# Inline CUDA implementation of HardTanh (clamp) with min / max parameters
# ---------------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void hardtanh_kernel(const scalar_t* __restrict__ input,
                                scalar_t* __restrict__ output,
                                scalar_t min_val,
                                scalar_t max_val,
                                int64_t numel) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    scalar_t val = input[idx];
    if (val < min_val)      val = min_val;
    else if (val > max_val) val = max_val;
    output[idx] = val;
}

torch::Tensor hardtanh_cuda(torch::Tensor input, double min_val, double max_val) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must reside on CUDA device");

    auto output = torch::empty_like(input);
    const int64_t numel   = input.numel();
    const int     threads = 256;
    const int     blocks  = (numel + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "hardtanh_cuda", ([&] {
        hardtanh_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            static_cast<scalar_t>(min_val),
            static_cast<scalar_t>(max_val),
            numel);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hardtanh_cuda", &hardtanh_cuda, "HardTanh CUDA kernel (clamp)");
}
"""

cpp_decl = "torch::Tensor hardtanh_cuda(torch::Tensor input, double min_val, double max_val);"

# Build/load the kernel
hardtanh_ext = load_inline(
    name="hardtanh_ext",
    cpp_sources=cpp_decl,
    cuda_sources=cuda_source,
    functions=["hardtanh_cuda"],
    verbose=False,
)


# ---------------------------------------------------------------------------
# Optimized model that swaps out PyTorch's HardTanh with the custom CUDA op
# ---------------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Optimized model:
        Linear -> GroupNorm -> custom CUDA HardTanh
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 num_groups: int,
                 hardtanh_min: float,
                 hardtanh_max: float):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max
        self._cuda_op = hardtanh_ext.hardtanh_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gemm(x)
        x = self.group_norm(x)
        # Use the custom CUDA op if the tensor is on GPU, else fall back to torch
        if x.is_cuda:
            x = self._cuda_op(x, self.hardtanh_min, self.hardtanh_max)
        else:
            x = torch.clamp(x, self.hardtanh_min, self.hardtanh_max)
        return x
