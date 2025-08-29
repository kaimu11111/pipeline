import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ----------------------------------------------------------------------
# hand-written CUDA kernels + host helpers
# ----------------------------------------------------------------------
source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

// ---------------------------------------------
// helpers : row-wise reductions inside a block
// ---------------------------------------------
template<int BLOCK_SIZE>
__device__ float block_reduce_max(float v){
    __shared__ float smem[BLOCK_SIZE];
    smem[threadIdx.x] = v;
    __syncthreads();
    for(int stride = BLOCK_SIZE/2; stride>0; stride >>= 1){
        if(threadIdx.x < stride){
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    return smem[0];
}

template<int BLOCK_SIZE>
__device__ float block_reduce_sum(float v){
    __shared__ float smem[BLOCK_SIZE];
    smem[threadIdx.x] = v;
    __syncthreads();
    for(int stride = BLOCK_SIZE/2; stride>0; stride >>= 1){
        if(threadIdx.x < stride){
            smem[threadIdx.x] += smem[threadIdx.x + stride];
        }
        __syncthreads();
    }
    return smem[0];
}

// ------------------------------------------------------
// Row-wise soft-max (2-D tensor, contiguous) – forward
// ------------------------------------------------------
template<int BLOCK_SIZE>
__global__ void rowwise_softmax_kernel(const float* __restrict__ in,
                                       float* __restrict__ out,
                                       const int rows,
                                       const int cols){
    int row = blockIdx.x;
    if(row >= rows) return;

    // compute max per row
    float local_max = -FLT_MAX;
    for(int idx = threadIdx.x; idx < cols; idx += BLOCK_SIZE){
        local_max = fmaxf(local_max, in[row*cols + idx]);
    }
    float row_max = block_reduce_max<BLOCK_SIZE>(local_max);

    // exponentiate & accumulate
    float local_sum = 0.f;
    for(int idx = threadIdx.x; idx < cols; idx += BLOCK_SIZE){
        float val = expf(in[row*cols + idx] - row_max);
        out[row*cols + idx] = val;
        local_sum += val;
    }
    float row_sum = block_reduce_sum<BLOCK_SIZE>(local_sum) + 1e-6f;

    // normalise
    for(int idx = threadIdx.x; idx < cols; idx += BLOCK_SIZE){
        out[row*cols + idx] /= row_sum;
    }
}

// ------------------------------------------------------
// Row-wise L2 normalisation  (2-D tensor, contiguous)
// ------------------------------------------------------
template<int BLOCK_SIZE>
__global__ void rowwise_l2norm_kernel(const float* __restrict__ in,
                                      float* __restrict__ out,
                                      const int rows,
                                      const int cols,
                                      const float eps){
    int row = blockIdx.x;
    if(row >= rows) return;

    // accumulate sum of squares
    float local_sum = 0.f;
    for(int idx = threadIdx.x; idx < cols; idx += BLOCK_SIZE){
        float v = in[row*cols + idx];
        local_sum += v*v;
    }
    float row_sum = block_reduce_sum<BLOCK_SIZE>(local_sum);
    float inv_norm = rsqrtf(row_sum + eps);

    // write normalised values
    for(int idx = threadIdx.x; idx < cols; idx += BLOCK_SIZE){
        out[row*cols + idx] = in[row*cols + idx] * inv_norm;
    }
}

// ------------------------------------------------------
// C++ functions callable from Python
// ------------------------------------------------------
torch::Tensor softmax_rowwise_cuda(torch::Tensor input){
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dim()==2 && input.is_contiguous(), "Expect contiguous 2-D tensor");

    const int rows = input.size(0);
    const int cols = input.size(1);
    auto output = torch::empty_like(input);

    constexpr int BLOCK = 256;
    rowwise_softmax_kernel<BLOCK><<<rows, BLOCK>>>(input.data_ptr<float>(),
                                                   output.data_ptr<float>(),
                                                   rows, cols);
    return output;
}

torch::Tensor l2norm_rowwise_cuda(torch::Tensor input, double eps){
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(input.dim()==2 && input.is_contiguous(), "Expect contiguous 2-D tensor");

    const int rows = input.size(0);
    const int cols = input.size(1);
    auto output = torch::empty_like(input);

    constexpr int BLOCK = 256;
    rowwise_l2norm_kernel<BLOCK><<<rows, BLOCK>>>(input.data_ptr<float>(),
                                                  output.data_ptr<float>(),
                                                  rows, cols,
                                                  static_cast<float>(eps));
    return output;
}
"""

# ----------------------------------------------------------------------
# C++ declarations seen by the auto-generated pybind wrapper
# ----------------------------------------------------------------------
cpp_src = """
torch::Tensor softmax_rowwise_cuda(torch::Tensor input);
torch::Tensor l2norm_rowwise_cuda(torch::Tensor input, double eps);
"""

# ----------------------------------------------------------------------
# build the extension
# ----------------------------------------------------------------------
ops = load_inline(
    name="vlad_custom_ops",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["softmax_rowwise_cuda", "l2norm_rowwise_cuda"],
    verbose=False,
)

# ----------------------------------------------------------------------
# PyTorch module that uses the custom kernels
# ----------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, cluster_size, feature_size, ghost_clusters):
        super().__init__()

        self.feature_size   = feature_size
        self.cluster_size   = cluster_size
        self.ghost_clusters = ghost_clusters

        init_sc    = 1.0 / (feature_size ** 0.5)
        clusters   = cluster_size + ghost_clusters

        self.clusters   = nn.Parameter(init_sc * torch.randn(feature_size, clusters))
        self.batch_norm = nn.BatchNorm1d(clusters)
        self.clusters2  = nn.Parameter(init_sc * torch.randn(1, feature_size, cluster_size))
        self.out_dim    = cluster_size * feature_size

        # expose CUDA kernels
        self.softmax_cuda = ops.softmax_rowwise_cuda
        self.l2norm_cuda  = ops.l2norm_rowwise_cuda

    def forward(self, x, mask=None):
        """
        Args:
            x: Tensor of shape (B, N, D)
        Returns:
            Tensor of shape (B, K*D)
        """
        B, N, D = x.shape
        x_flat = x.reshape(-1, self.feature_size)               # (B·N, D)

        assignment = torch.matmul(x_flat, self.clusters)        # (B·N, K+G)
        assignment = self.batch_norm(assignment)

        # row-wise soft-max (custom kernel)
        assignment = self.softmax_cuda(assignment.contiguous()) # (B·N, K+G)

        assignment = assignment[:, :self.cluster_size]          # drop ghosts
        assignment = assignment.view(B, N, self.cluster_size)   # (B, N, K)
        a_sum      = assignment.sum(dim=1, keepdim=True)        # (B, 1, K)
        a          = a_sum * self.clusters2                     # (B, D, K)

        assignment = assignment.transpose(1, 2)                 # (B, K, N)
        x_reshaped = x.reshape(B, N, D)                         # (B, N, D)

        vlad = torch.matmul(assignment, x_reshaped)             # (B, K, D)
        vlad = vlad.transpose(1, 2)                             # (B, D, K)
        vlad = vlad - a

        vlad = torch.nn.functional.normalize(vlad, p=2, dim=1)  # intra-cluster
        vlad = vlad.reshape(B, -1)                              # (B, K·D)

        # global L2 normalisation (custom kernel)
        vlad = self.l2norm_cuda(vlad.contiguous(), 1e-12)

        return vlad
