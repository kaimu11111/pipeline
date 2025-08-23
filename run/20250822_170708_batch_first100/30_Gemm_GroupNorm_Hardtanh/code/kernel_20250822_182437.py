import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


# -------------------------------------------------------------------------
# Inline CUDA/C++ source for custom operators:
# 1) Naive GEMM
# 2) Naive GroupNorm
# 3) Naive HardTanh
# (These are purely illustrative "hand-written" kernels, not optimized!)
# -------------------------------------------------------------------------
source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

static const int BLOCK_SIZE = 256;

// ----------------------------------------
// 1) Naive GEMM
//    x: [N, in_features], W: [out_features, in_features], bias: [out_features]
//    out: [N, out_features]
// ----------------------------------------
__global__ void gemm_naive_kernel(const float* __restrict__ x,
                                  const float* __restrict__ W,
                                  const float* __restrict__ bias,
                                  float* __restrict__ out,
                                  int N, int in_features, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * out_features) {
        int n = idx / out_features;
        int o = idx % out_features;
        float val = bias[o];
        for(int i = 0; i < in_features; i++){
            val += x[n * in_features + i] * W[o * in_features + i];
        }
        out[idx] = val;
    }
}

torch::Tensor gemm_naive_cuda(torch::Tensor x,
                              torch::Tensor W,
                              torch::Tensor bias) {
    // Assumes x is [N, in_features], W is [out_features, in_features]
    // Ensure contiguous
    x = x.contiguous();
    W = W.contiguous();
    bias = bias.contiguous();

    auto N = x.size(0);
    auto in_features = x.size(1);
    auto out_features = W.size(0);

    auto out = torch::empty({(long)N, (long)out_features}, x.options());

    int totalThreads = N * out_features;
    int numBlocks = (totalThreads + BLOCK_SIZE - 1) / BLOCK_SIZE;

    gemm_naive_kernel<<<numBlocks, BLOCK_SIZE>>>(
        x.data_ptr<float>(),
        W.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        N, in_features, out_features
    );

    return out;
}

// ---------------------------------------------------------------
// 2) Naive GroupNorm (two-kernel approach)
//    x, gamma, beta, out all [N, C], split into num_groups
//    We'll compute per-group mean & var, then apply them.
// ---------------------------------------------------------------
__global__ void compute_mean_var_kernel(const float* __restrict__ x,
                                        float* __restrict__ mean,
                                        float* __restrict__ var,
                                        int N, int C, int G, int group_size,
                                        float eps) {
    // Each block corresponds to one (n, g) pair
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * G) {
        int n = idx / G;
        int g = idx % G;
        int start_c = g * group_size;
        int end_c = start_c + group_size;

        // Compute mean
        float sum_val = 0.0f;
        for(int c = start_c; c < end_c; c++){
            sum_val += x[n * C + c];
        }
        float mean_val = sum_val / (float)group_size;

        // Compute variance
        float sum_sq = 0.0f;
        for(int c = start_c; c < end_c; c++){
            float diff = x[n * C + c] - mean_val;
            sum_sq += diff * diff;
        }
        float var_val = sum_sq / (float)group_size;

        mean[idx] = mean_val;
        var[idx]  = var_val + eps; // store eps-added var
    }
}

__global__ void apply_groupnorm_kernel(const float* __restrict__ x,
                                       const float* __restrict__ mean,
                                       const float* __restrict__ var,
                                       const float* __restrict__ gamma,
                                       const float* __restrict__ beta,
                                       float* __restrict__ out,
                                       int N, int C, int G, int group_size) {
    // Each thread -> 1 element in x: total is N*C
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * C) {
        int n = idx / C;
        int c = idx % C;
        int g = c / group_size;  // which group does channel c belong to
        float mean_val = mean[n * G + g];
        float var_val  = var[n * G + g];
        float inv_std  = rsqrtf(var_val);

        float x_normed = (x[idx] - mean_val) * inv_std;
        float result   = x_normed * gamma[c] + beta[c];
        out[idx] = result;
    }
}

torch::Tensor groupnorm_naive_cuda(torch::Tensor x,
                                   torch::Tensor gamma,
                                   torch::Tensor beta,
                                   int num_groups,
                                   float eps) {
    x = x.contiguous();
    gamma = gamma.contiguous();
    beta = beta.contiguous();

    auto N = x.size(0);
    auto C = x.size(1);

    auto out = torch::empty_like(x);

    int G = num_groups;
    int group_size = C / G;

    // Temporary buffers for mean & var: [N, G]
    auto mean = torch::zeros({(long)N, (long)G}, x.options());
    auto var  = torch::zeros({(long)N, (long)G}, x.options());

    // kernel 1: compute mean/var
    int totalGroups = N * G;
    int numBlocks = (totalGroups + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_mean_var_kernel<<<numBlocks, BLOCK_SIZE>>>(
        x.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        N, C, G, group_size, eps
    );

    // kernel 2: apply groupnorm
    int totalElems = N * C;
    numBlocks = (totalElems + BLOCK_SIZE - 1) / BLOCK_SIZE;
    apply_groupnorm_kernel<<<numBlocks, BLOCK_SIZE>>>(
        x.data_ptr<float>(),
        mean.data_ptr<float>(),
        var.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, G, group_size
    );

    return out;
}

// -----------------------------------------------
// 3) Naive HardTanh
//    y = clamp(x, min_val, max_val)
// -----------------------------------------------
__global__ void hardtanh_naive_kernel(const float* __restrict__ x,
                                      float* __restrict__ out,
                                      float min_val, float max_val,
                                      int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        if (val < min_val) val = min_val;
        if (val > max_val) val = max_val;
        out[idx] = val;
    }
}

torch::Tensor hardtanh_naive_cuda(torch::Tensor x,
                                  float min_val,
                                  float max_val) {
    x = x.contiguous();
    auto out = torch::empty_like(x);

    int size = x.numel();
    int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    hardtanh_naive_kernel<<<numBlocks, BLOCK_SIZE>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        min_val, max_val, size
    );

    return out;
}

// --------------------------------------------------------------------
// PyBind
// --------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_naive_cuda", &gemm_naive_cuda, "Naive GEMM");
    m.def("groupnorm_naive_cuda", &groupnorm_naive_cuda, "Naive GroupNorm");
    m.def("hardtanh_naive_cuda", &hardtanh_naive_cuda, "Naive HardTanh");
}
""".strip()


cpp_src = r"""
torch::Tensor gemm_naive_cuda(torch::Tensor x,
                              torch::Tensor W,
                              torch::Tensor bias);

torch::Tensor groupnorm_naive_cuda(torch::Tensor x,
                                   torch::Tensor gamma,
                                   torch::Tensor beta,
                                   int num_groups,
                                   float eps);

torch::Tensor hardtanh_naive_cuda(torch::Tensor x,
                                  float min_val,
                                  float max_val);
""".strip()

# Build the custom module
model_optim = load_inline(
    name="model_optim",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=[
        "gemm_naive_cuda",
        "groupnorm_naive_cuda",
        "hardtanh_naive_cuda",
    ],
    verbose=False,
)


# -------------------------------------------------------------------------
# New Model Definition with custom kernels
# -------------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Model that replaces
    1) GEMM + 2) GroupNorm + 3) HardTanh
    with naive, hand-written CUDA kernels for demonstration.
    """
    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        # Replicate linear (gemm) parameters
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Replicate groupnorm parameters
        self.gamma = nn.Parameter(torch.ones(out_features))
        self.beta = nn.Parameter(torch.zeros(out_features))
        self.num_groups = num_groups
        self.eps = 1e-5

        # Save HardTanh bounds
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

    def forward(self, x):
        # 1) GEMM
        out = model_optim.gemm_naive_cuda(
            x, self.weight, self.bias
        )
        # 2) GroupNorm
        out = model_optim.groupnorm_naive_cuda(
            out, self.gamma, self.beta, self.num_groups, self.eps
        )
        # 3) HardTanh
        out = model_optim.hardtanh_naive_cuda(
            out, self.hardtanh_min, self.hardtanh_max
        )
        return out


# -------------------------------------------------------------------------
# Utility functions to match the signature of the original code
# -------------------------------------------------------------------------
batch_size = 1024
in_features = 8192
out_features = 8192
num_groups = 16
hardtanh_min = -2.0
hardtanh_max = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features, num_groups, hardtanh_min, hardtanh_max]
