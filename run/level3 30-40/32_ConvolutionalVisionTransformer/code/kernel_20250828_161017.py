# 1. Imports ───────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# 2. CUDA source (kernel + host wrapper) ───────────────────────────────
source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------
__global__ void prepend_cls_kernel(
        const float* __restrict__ inp,     // (B, D)
        const float* __restrict__ cls,     // (D)
        float* __restrict__ out,           // (B, 2, D)
        const int B,
        const int D) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * D) return;

    const int b = idx / D;          // batch index
    const int d = idx % D;          // feature index

    // CLS token is shared across all samples
    out[b * 2 * D + d]       = cls[d];
    // Per-sample embedding
    out[b * 2 * D + D + d]   = inp[idx];
}

// ---------------------------------------------------------------------
// Host wrapper
// ---------------------------------------------------------------------
torch::Tensor prepend_cls_token_cuda(torch::Tensor inp,
                                     torch::Tensor cls) {
    TORCH_CHECK(inp.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(cls.is_cuda(), "CLS token must be on CUDA");
    TORCH_CHECK(inp.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(cls.dtype()  == torch::kFloat32, "CLS token must be float32");
    TORCH_CHECK(cls.numel() == inp.size(1),
                "CLS token dimension mismatch");

    const int B = inp.size(0);
    const int D = inp.size(1);

    auto out = torch::empty({B, 2, D}, inp.options());

    const int threads = 256;
    const int blocks  = (B * D + threads - 1) / threads;

    prepend_cls_kernel<<<blocks, threads>>>(
        inp.data_ptr<float>(),
        cls.data_ptr<float>(),
        out.data_ptr<float>(),
        B, D);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));

    return out;
}
"""

# 3. C++ prototypes ────────────────────────────────────────────────────
cpp_src = """
torch::Tensor prepend_cls_token_cuda(torch::Tensor inp,
                                     torch::Tensor cls);
"""

# 4. Build / load the extension ────────────────────────────────────────
prepend_cls_token_ext = load_inline(
    name         = "prepend_cls_token_ext",
    cpp_sources  = cpp_src,
    cuda_sources = source,
    functions    = ["prepend_cls_token_cuda"],
    verbose      = False
)

# 5. Model definition ─────────────────────────────────────────────────
class ModelNew(nn.Module):
    """
    Convolutional Vision Transformer with a custom CUDA kernel that
    prepends a CLS token (with full autograd support).
    """

    class _PrependCLSTokenFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inp, cls):
            out = prepend_cls_token_ext.prepend_cls_token_cuda(
                inp.contiguous(), cls.contiguous())
            ctx.save_for_backward(inp)
            return out

        @staticmethod
        def backward(ctx, grad_out):
            (inp,) = ctx.saved_tensors
            grad_inp = grad_out[:, 1, :].contiguous()            # gradient w.r.t. sample embeddings
            grad_cls = grad_out[:, 0, :].sum(dim=0).contiguous() # gradient w.r.t. shared CLS token
            return grad_inp, grad_cls

    # -----------------------------------------------------------------
    def __init__(self, num_classes,
                 embed_dim    = 512,
                 num_heads    = 8,
                 num_layers   = 6,
                 mlp_ratio    = 4.0,
                 patch_size   = 4,
                 in_channels  = 3):
        super().__init__()

        self.embed_dim  = embed_dim
        self.patch_size = patch_size

        # Patch embedding (convolutional)
        self.conv1 = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size = patch_size,
            stride      = patch_size
        )

        # Linear projection from flattened patches to embed_dim.
        # Use LazyLinear so the in_features dimension is inferred
        # on the first forward pass (handles any image size).
        self.linear_proj = nn.LazyLinear(embed_dim)

        # Transformer encoder stack
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model         = embed_dim,
                nhead           = num_heads,
                dim_feedforward = int(embed_dim * mlp_ratio),
                dropout         = 0.0,
                batch_first     = True
            ) for _ in range(num_layers)
        ])

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Classification head
        self.fc_out = nn.Linear(embed_dim, num_classes)

    # -----------------------------------------------------------------
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.conv1(x)                          # (B, D, H', W')
        B, D, H, W = x.shape
        x = x.reshape(B, D * H * W)                # Flatten patches → (B, D*H'*W')
        x = self.linear_proj(x)                    # Project to embed_dim → (B, D)

        # Prepend CLS token via custom CUDA kernel
        cls_tok = self.cls_token.view(-1)          # (D,)
        x = self._PrependCLSTokenFunction.apply(x, cls_tok)  # (B, 2, D)

        # Transformer encoder
        for layer in self.transformer_layers:
            x = layer(x)

        # Classification head (use CLS token)
        return self.fc_out(x[:, 0])
