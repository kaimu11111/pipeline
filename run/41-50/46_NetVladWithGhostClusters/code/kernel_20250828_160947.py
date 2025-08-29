import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline


# ------------------------------------------------------------------
# CUDA kernel : fast row-wise L2-normalisation (F.normalize, dim=1)
# ------------------------------------------------------------------
cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__global__ void l2_normalise_kernel(const scalar_t* __restrict__ inp,
                                    scalar_t* __restrict__ out,
                                    const int row_size,
                                    const float eps)
{
    const int row = blockIdx.x;                 // one block per row
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    // (1) compute squared-L2 for the row
    float local_sum = 0.f;
    for (int i = tid; i < row_size; i += stride) {
        float v = static_cast<float>(inp[row * row_size + i]);
        local_sum += v * v;
    }

    __shared__ float red_sum;
    if (tid == 0) red_sum = 0.f;
    __syncthreads();

    atomicAdd(&red_sum, local_sum);
    __syncthreads();

    const float inv_norm = rsqrtf(red_sum + eps);

    // (2) write back normalised values
    for (int i = tid; i < row_size; i += stride) {
        out[row * row_size + i] =
            static_cast<scalar_t>(static_cast<float>(inp[row * row_size + i]) * inv_norm);
    }
}

torch::Tensor l2_normalise_cuda(torch::Tensor x, double eps)
{
    TORCH_CHECK(x.is_cuda(), "input must reside on CUDA device");
    TORCH_CHECK(x.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32,
                "only float32 tensors are supported");

    const int rows = x.size(0);
    const int row_size = x.size(1);

    auto out = torch::empty_like(x);

    const int threads = 256;
    const dim3 blocks(rows);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "l2_normalise_cuda", ([&]{
        l2_normalise_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            row_size,
            static_cast<float>(eps)
        );
    }));
    return out;
}
"""

cpp_decl = "torch::Tensor l2_normalise_cuda(torch::Tensor x, double eps);"

# compile and load
l2_norm_mod = load_inline(
    name="l2_norm_cuda",
    cpp_sources=cpp_decl,
    cuda_sources=cuda_src,
    functions=["l2_normalise_cuda"],
    verbose=False
)


# ------------------------------------------------------------------
# Optimised Model
# ------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, cluster_size, feature_size, ghost_clusters):
        super().__init__()

        self.cluster_size = cluster_size
        self.feature_size = feature_size
        self.ghost_clusters = ghost_clusters

        init_sc = 1.0 / math.sqrt(feature_size)
        total_clusters = cluster_size + ghost_clusters

        # parameters
        self.clusters = nn.Parameter(init_sc * torch.randn(feature_size, total_clusters))
        self.batch_norm = nn.BatchNorm1d(total_clusters)
        self.clusters2 = nn.Parameter(init_sc * torch.randn(1, feature_size, cluster_size))

        # expose custom cuda op
        self.l2_norm = l2_norm_mod.l2_normalise_cuda

        self.out_dim = cluster_size * feature_size

    def _row_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Apply fast L2 normalisation along dim=1 (expects contiguous 2-D tensor)."""
        return self.l2_norm(x, 1e-12)

    def forward(self, x: torch.Tensor, mask=None):
        """
        Args:
            x : (B, N, D)
        Returns:
            (B, D*K)
        """
        B, N, D = x.shape
        x_flat = x.reshape(-1, D)                           # (B*N, D)

        # soft-assignment
        assignment = torch.matmul(x_flat, self.clusters)    # (B*N, K+G)
        assignment = self.batch_norm(assignment)
        assignment = F.softmax(assignment, dim=1)[:, :self.cluster_size]
        assignment = assignment.view(B, N, self.cluster_size)

        # compute residuals
        a_sum = assignment.sum(dim=1, keepdim=True)         # (B, 1, K)
        a = a_sum * self.clusters2                          # (B, D, K)

        assignment = assignment.transpose(1, 2)             # (B, K, N)
        x_exp = x_flat.view(B, N, D)                        # (B, N, D)

        vlad = torch.matmul(assignment, x_exp)              # (B, K, D)
        vlad = vlad.transpose(1, 2)                         # (B, D, K)
        vlad = vlad - a                                     # (B, D, K)

        # intra-normalisation (dim=1 -> over D)
        vlad = vlad.permute(0, 2, 1).contiguous()           # (B, K, D)
        vlad = self._row_norm(vlad.view(-1, D))             # (B*K, D)
        vlad = vlad.view(B, self.cluster_size, D).permute(0, 2, 1).contiguous()

        # flatten  & final L2
        vlad = vlad.view(B, -1)                             # (B, D*K)
        vlad = self._row_norm(vlad)                         # (B, D*K)
        return vlad


# ------------------------------------------------------------------
# Helper functions for external usage --------------------------------
# ------------------------------------------------------------------
batch_size = 1024
num_features = 50
num_clusters = 16
feature_size = 256
ghost_clusters = 8


def get_inputs():
    return [torch.rand(batch_size, num_features, feature_size, device="cuda")]


def get_init_inputs():
    return [num_clusters, feature_size, ghost_clusters]
