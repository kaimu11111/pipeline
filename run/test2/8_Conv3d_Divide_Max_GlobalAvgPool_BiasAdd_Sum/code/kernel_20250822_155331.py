import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------------
# 1. CUDA / C++ source code -----------------------------------------
# ------------------------------------------------------------------
source = r"""
#include <torch/extension.h>
#include <ATen/ATen.h>

// ------------------------------------------------------------
// Helper: SAME-padding conv3d (stride=1, dilation=1, groups=1)
// ------------------------------------------------------------
torch::Tensor conv3d_same_cuda(torch::Tensor input,
                               torch::Tensor weight) {
    // Expect input  (N,C_in,D,H,W)
    //        weight (C_out,C_in,kd,kh,kw)
    const int64_t kd = weight.size(2);
    const int64_t kh = weight.size(3);
    const int64_t kw = weight.size(4);

    std::vector<int64_t> padding = {kd / 2, kh / 2, kw / 2};

    return at::conv3d(
        input,
        weight,
        /*bias=*/{},
        /*stride=*/{1, 1, 1},
        padding,
        /*dilation=*/{1, 1, 1},
        /*groups=*/1);
}

torch::Tensor scale_const_cuda(torch::Tensor input, double factor) {
    return input * factor;
}

torch::Tensor maxpool3d_cuda(torch::Tensor input,
                             std::vector<int64_t> kernel_size) {
    return at::max_pool3d(
        input,
        kernel_size,
        /*stride=*/kernel_size,
        /*padding=*/{0, 0, 0},
        /*dilation=*/{1, 1, 1},
        /*ceil_mode=*/false);
}

torch::Tensor global_avg_pool3d_cuda(torch::Tensor input) {
    // mean over D,H,W, keep C dimension intact
    return input.mean({2, 3, 4}, /*keepdim=*/true);
}

torch::Tensor add_bias_cuda(torch::Tensor input,
                            torch::Tensor bias) {
    // bias shape: (C,)
    return input + bias.view({1, -1, 1, 1, 1});
}

torch::Tensor sum_dim1_cuda(torch::Tensor input) {
    return input.sum(1);
}

// ------------------------- Python bindings ------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv3d_same_cuda",      &conv3d_same_cuda,       "SAME-padding conv3d (CUDA)");
    m.def("scale_const_cuda",      &scale_const_cuda,       "scale_const (CUDA)");
    m.def("maxpool3d_cuda",        &maxpool3d_cuda,         "maxpool3d (CUDA)");
    m.def("global_avg_pool3d_cuda",&global_avg_pool3d_cuda, "gap3d (CUDA)");
    m.def("add_bias_cuda",         &add_bias_cuda,          "add_bias (CUDA)");
    m.def("sum_dim1_cuda",         &sum_dim1_cuda,          "sum_dim1 (CUDA)");
}
"""

# ------------------------------------------------------------------
# 2. C++ prototypes -------------------------------------------------
# ------------------------------------------------------------------
cpp_src = """
torch::Tensor conv3d_same_cuda(torch::Tensor input, torch::Tensor weight);
torch::Tensor scale_const_cuda(torch::Tensor input, double factor);
torch::Tensor maxpool3d_cuda(torch::Tensor input, std::vector<int64_t> kernel_size);
torch::Tensor global_avg_pool3d_cuda(torch::Tensor input);
torch::Tensor add_bias_cuda(torch::Tensor input, torch::Tensor bias);
torch::Tensor sum_dim1_cuda(torch::Tensor input);
"""

# ------------------------------------------------------------------
# 3. Compilation ----------------------------------------------------
# ------------------------------------------------------------------
kernels = load_inline(
    name="custom_cuda_kernels_fixed_v5",
    cpp_sources=cpp_src,
    cuda_sources=source,
    verbose=False,
)

# ------------------------------------------------------------------
# 4. Python wrapper Model ------------------------------------------
# ------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    CUDA-accelerated model that mirrors original behaviour
    using custom extension functions.
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size, factor, pool_size,
                 bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        kd, kh, kw = kernel_size
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kd, kh, kw,
                        device='cuda', dtype=torch.float32))
        self.bias   = nn.Parameter(
            torch.randn(*bias_shape, device='cuda', dtype=torch.float32))
        self.factor     = float(factor)
        self.pool_size  = list(pool_size)
        self.sum_dim    = sum_dim  # expected 1

    def forward(self, x):
        x = x.contiguous().float()
        x = kernels.conv3d_same_cuda(x, self.weight)
        x = kernels.scale_const_cuda(x, self.factor)
        x = kernels.maxpool3d_cuda(x, self.pool_size)
        x = kernels.global_avg_pool3d_cuda(x)
        x = kernels.add_bias_cuda(x, self.bias)
        if self.sum_dim == 1:
            x = kernels.sum_dim1_cuda(x)
        else:
            raise NotImplementedError("Only sum_dim==1 supported")
        return x
