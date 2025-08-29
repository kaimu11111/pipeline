import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------------
# Custom CUDA kernel: prepend_cls_token
# This kernel creates an output tensor of shape (B, 2, D)
# where the first token along dim=1 is the shared cls token
# and the second token is the per-sample embedding vector.
# ---------------------------------------------------------------------------
cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

////////////////////////////////////////////////////////////////////////////////
// Kernel
////////////////////////////////////////////////////////////////////////////////
__global__ void prepend_cls_kernel(
        const float* __restrict__ inp,          // (B, D)
        const float* __restrict__ cls,          // (D)
        float* __restrict__ out,                // (B, 2, D)
        const int B,
        const int D) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * D) return;

    const int b = idx / D;      // batch index
    const int d = idx % D;      // feature index

    // write cls token
    out[b * 2 * D + d]       = cls[d];
    // write input embedding as second token
    out[b * 2 * D + D + d]   = inp[idx];
}

////////////////////////////////////////////////////////////////////////////////
// Interface
////////////////////////////////////////////////////////////////////////////////
torch::Tensor prepend_cls_token_cuda(torch::Tensor inp, torch::Tensor cls) {
    TORCH_CHECK(inp.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(cls.is_cuda(), "CLS token must be a CUDA tensor");
    TORCH_CHECK(inp.dtype() == torch::kFloat32,
                "Only float32 tensors are supported");
    TORCH_CHECK(cls.dtype() == torch::kFloat32,
                "Only float32 tensors are supported");

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("prepend_cls_token_cuda", &prepend_cls_token_cuda,
          "Prepend CLS token (CUDA)");
}
"""

cpp_decl = """
torch::Tensor prepend_cls_token_cuda(torch::Tensor inp, torch::Tensor cls);
"""

# Compile the CUDA extension (done once at import time)
prepend_cls = load_inline(
    name="prepend_cls_token_ext",
    cpp_sources=cpp_decl,
    cuda_sources=cuda_src,
    functions=["prepend_cls_token_cuda"],
    verbose=False,
)

# ---------------------------------------------------------------------------
# Optimised model using the custom CUDA kernel
# ---------------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, num_classes, embed_dim=512, num_heads=8, num_layers=6,
                 mlp_ratio=4.0, patch_size=4, in_channels=3, image_size=32):
        super().__init__()

        self.patch_size  = patch_size
        self.image_size  = image_size
        self.embed_dim   = embed_dim

        # patch embedding
        self.conv1 = nn.Conv2d(in_channels, embed_dim,
                               kernel_size=patch_size, stride=patch_size)

        num_patches = (image_size // patch_size) ** 2
        self.linear_proj = nn.Linear(embed_dim * num_patches, embed_dim)

        # transformer encoder stack
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=0.0,
                batch_first=True
            ) for _ in range(num_layers)
        ])

        # cls token parameter
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # classification head
        self.fc_out = nn.Linear(embed_dim, num_classes)

        # save reference to compiled extension
        self._prepend_cls = prepend_cls

    def forward(self, x):
        """
        x : Tensor, shape = (B, C, H, W)
        """
        B = x.size(0)

        # patch embedding
        x = self.conv1(x)                          # (B, D, H', W')
        x = x.flatten(start_dim=1)                 # (B, D * num_patches)
        x = self.linear_proj(x)                    # (B, D)

        # call fused CUDA kernel to build sequence with CLS token
        # cls_token: (D)
        cls_tok = self.cls_token.view(-1).contiguous()
        x = self._prepend_cls.prepend_cls_token_cuda(
                x.contiguous(), cls_tok)           # (B, 2, D)

        # transformer encoder
        for layer in self.transformer_layers:
            x = layer(x)

        # classification head (use CLS token)
        return self.fc_out(x[:, 0])

# === Helper functions (unchanged) ===
batch_size  = 5
image_size  = 16
embed_dim   = 64
in_channels = 3
num_heads   = 4
num_classes = 100

def get_inputs():
    return [torch.rand(batch_size, in_channels, image_size, image_size, device='cuda')]

def get_init_inputs():
    return [num_classes, embed_dim, num_heads]
