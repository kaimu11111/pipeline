import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

template<int TPB>
__global__ void fused_add_ln_kernel(
        const float* __restrict__ a,
        const float* __restrict__ b,
        const float* __restrict__ gamma,
        const float* __restrict__ beta,
        float* __restrict__ out,
        int rows,
        int cols,
        float eps)
{
    extern __shared__ float shared_mem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;

    // pointer to the first element of this row
    int base = row * cols;

    // each thread processes one column element if in range
    float val = 0.0f;
    if (tid < cols) {
        val = a[base + tid] + b[base + tid];   // residual connection + addition
    }
    shared_mem[tid] = (tid < cols) ? val : 0.0f;
    __syncthreads();

    // ----- compute mean -----
    float sum = shared_mem[tid];
    // reduction in shared memory
    for (int stride = TPB >> 1; stride > 0; stride >>= 1) {
        __syncthreads();
        if (tid < stride)
            shared_mem[tid] += shared_mem[tid + stride];
    }
    float mean = shared_mem[0] / static_cast<float>(cols);

    // ----- compute variance -----
    float diff = (tid < cols) ? (val - mean) : 0.0f;
    shared_mem[tid] = diff * diff;
    __syncthreads();
    for (int stride = TPB >> 1; stride > 0; stride >>= 1) {
        __syncthreads();
        if (tid < stride)
            shared_mem[tid] += shared_mem[tid + stride];
    }
    float var = shared_mem[0] / static_cast<float>(cols);
    float inv_std = rsqrtf(var + eps);

    // ----- normalize, scale, shift -----
    if (tid < cols) {
        float norm = (val - mean) * inv_std;
        out[base + tid] = norm * gamma[tid] + beta[tid];
    }
}

torch::Tensor fused_add_layernorm_cuda(torch::Tensor a,
                                       torch::Tensor b,
                                       torch::Tensor gamma,
                                       torch::Tensor beta,
                                       double eps) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "Input tensors must be CUDA tensors");
    TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "Only float32 is supported");

    auto out = torch::empty_like(a);
    int seq_len = a.size(0);
    int batch = a.size(1);
    int rows = seq_len * batch;          // flatten the first two dims
    int cols = a.size(2);               // embedding dimension

    const int TPB = 1024;               // threads per block
    int tpb = 1;
    // choose tpb as power-of-two >= cols and <=1024
    while (tpb < cols) tpb <<= 1;
    if (tpb > TPB) tpb = TPB;

    dim3 block(tpb);
    dim3 grid(rows);
    size_t shm = tpb * sizeof(float);

    fused_add_ln_kernel<1024><<<grid, block, shm>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        out.data_ptr<float>(),
        rows,
        cols,
        static_cast<float>(eps)
    );
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_add_layernorm_cuda", &fused_add_layernorm_cuda, "Fused Add+LayerNorm (CUDA)");
}
"""

CPP_DECL = "torch::Tensor fused_add_layernorm_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor gamma, torch::Tensor beta, double eps);"

fused_add_ln = load_inline(
    name="fused_add_layernorm",
    cpp_sources=CPP_DECL,
    cuda_sources=CUDA_SRC,
    functions=["fused_add_layernorm_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        :param x: Input tensor of shape (B, C, H, W)
        :return: Tensor of the same shape after MHA + residual + LayerNorm
        """
        B, C, H, W = x.shape
        x_reshaped = x.view(B, C, H * W).permute(2, 0, 1)  # (seq_len, B, C)

        attn_output, _ = self.attn(x_reshaped, x_reshaped, x_reshaped)

        # Fused residual connection + LayerNorm
        y = fused_add_ln.fused_add_layernorm_cuda(
            attn_output.contiguous(),
            x_reshaped.contiguous(),
            self.norm.weight.contiguous(),
            self.norm.bias.contiguous(),
            self.norm.eps,
        )

        y = y.permute(1, 2, 0).view(B, C, H, W)
        return y


# Maintain the same helper functions
embed_dim = 64
num_heads = 4
batch_size = 2
num_channels = embed_dim
image_height = 64
image_width = 64


def get_inputs():
    return [torch.rand(batch_size, num_channels, image_height, image_width).cuda()]


def get_init_inputs():
    return [embed_dim, num_heads]
