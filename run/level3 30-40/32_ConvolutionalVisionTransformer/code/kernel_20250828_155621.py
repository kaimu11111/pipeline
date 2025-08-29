import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------------
# CUDA source (kernel + host wrapper)
# ---------------------------------------------------------------------------
source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
__global__ void prepend_cls_kernel(
        const float* __restrict__ inp,     // (B, D)
        const float* __restrict__ cls,     // (D)
        float* __restrict__ out,           // (B, 2, D)
        const int B,
        const int D) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * D) return;

    const int b = idx / D;    // batch index
    const int d = idx % D;    // feature index

    // first token (shared CLS)
    out[b * 2 * D + d]     = cls[d];
    // second token (per-sample embedding)
    out[b * 2 * D + D + d] = inp[idx];
}

// ---------------------------------------------------------------------------
// Host wrapper
// ---------------------------------------------------------------------------
torch::Tensor prepend_cls_token_cuda(torch::Tensor inp, torch::Tensor cls) {
    TORCH_CHECK(inp.is_cuda(), "Input must reside on CUDA");
    TORCH_CHECK(cls.is_cuda(), "CLS token must reside on CUDA");
    TORCH_CHECK(inp.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(cls.dtype() == torch::kFloat32, "CLS token must be float32");
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

    return out;
}
"""

# ---------------------------------------------------------------------------
# C++ prototypes
# ---------------------------------------------------------------------------
cpp_src = """
torch::Tensor prepend_cls_token_cuda(torch::Tensor inp, torch::Tensor cls);
"""

# ---------------------------------------------------------------------------
# Build / load the extension
# ---------------------------------------------------------------------------
prepend_cls_token_ext = load_inline(
    name="prepend_cls_token_ext",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["prepend_cls_token_cuda"],
    verbose=False,
)

# ---------------------------------------------------------------------------
# PyTorch module that uses the custom kernel
# ---------------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, num_classes, embed_dim=512, num_heads=8, num_layers=6,
                 mlp_ratio=4.0, patch_size=4, in_channels=3, image_size=32):
        super().__init__()

        self.patch_size = patch_size
        self.image_size = image_size
        self.embed_dim = embed_dim

        # Patch embedding
        self.conv1 = nn.Conv2d(in_channels, embed_dim,
                               kernel_size=patch_size, stride=patch_size)

        num_patches = (image_size // patch_size) ** 2
        self.linear_proj = nn.Linear(embed_dim * num_patches, embed_dim)

        # Transformer encoder stack
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=0.0,
                batch_first=True
            ) for _ in range(num_layers)
        ])

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Classification head
        self.fc_out = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.conv1(x)              # (B, D, H', W')
        x = x.flatten(start_dim=1)     # (B, D * num_patches)
        x = self.linear_proj(x)        # (B, D)

        # Prepend CLS token using custom CUDA kernel
        cls_tok = self.cls_token.view(-1).contiguous()   # (D,)
        x = prepend_cls_token_ext.prepend_cls_token_cuda(
            x.contiguous(), cls_tok)                     # (B, 2, D)

        # Transformer encoder
        for layer in self.transformer_layers:
            x = layer(x)

        # Classification head (use CLS token)
        return self.fc_out(x[:, 0])
