import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------------
# CUDA / C++ source: high-accuracy implementation using cuBLAS
# ------------------------------------------------------------------
source = r"""
#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/*
 * High-accuracy forward pass implemented with cuBLAS via at::matmul.
 *
 * Computes:
 *   Y = X · Wᵀ + bias
 *
 *   X : (B, inF)   row-major
 *   W : (outF, inF)
 *   Y : (B, outF)
 */
torch::Tensor model_fwd_cuda(torch::Tensor X,
                             torch::Tensor W,
                             torch::Tensor bias) {
    CHECK_INPUT(X);
    CHECK_INPUT(W);
    CHECK_INPUT(bias);

    TORCH_CHECK(X.scalar_type() == torch::kFloat32, "X must be float32");
    TORCH_CHECK(W.scalar_type() == torch::kFloat32, "W must be float32");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "bias must be float32");

    TORCH_CHECK(W.size(1) == X.size(1),        "W shape mismatch");
    TORCH_CHECK(bias.numel() == W.size(0),     "bias shape mismatch");

    // Y = X @ W.T
    auto Y = torch::matmul(X, W.t());          // (B, outF)
    Y = Y + bias;                              // broadcast add
    return Y;
}
"""

# ------------------------------------------------------------------
# C++ declaration to expose
# ------------------------------------------------------------------
cpp_src = """
torch::Tensor model_fwd_cuda(torch::Tensor X,
                             torch::Tensor W,
                             torch::Tensor bias);
"""

# ------------------------------------------------------------------
# Compile & load
# ------------------------------------------------------------------
model_cuda = load_inline(
    name="model_cuda_kernels_fixed",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["model_fwd_cuda"],
    verbose=False,
)

# ------------------------------------------------------------------
# PyTorch module wrapping the CUDA implementation
# ------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True).cuda()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            x = x.cuda(non_blocking=True)
        x = x.contiguous()
        y = model_cuda.model_fwd_cuda(
            x,
            self.linear.weight.contiguous(),  # (outF, inF)
            self.linear.bias.contiguous(),
        )
        return y
