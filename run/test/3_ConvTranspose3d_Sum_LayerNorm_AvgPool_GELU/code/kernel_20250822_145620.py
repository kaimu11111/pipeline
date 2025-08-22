import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------------
# CUDA source with all kernels and C++/CUDA host wrappers
# ------------------------------------------------------------------
source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <vector>

////////////////////////////////////////////////////////////////
// Helpers
////////////////////////////////////////////////////////////////
#define CUDA_1D_KERNEL_LOOP(i, n)                                   \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);      \
       i += blockDim.x * gridDim.x)

inline int GET_BLOCKS(const int N)
{
    const int kThreadsPerBlock = 256;
    return (N + kThreadsPerBlock - 1) / kThreadsPerBlock;
}

////////////////////////////////////////////////////////////////
// Conv-Transpose-3D (naïve scatter + atomics) – fixed kernel flip
////////////////////////////////////////////////////////////////
__global__ void conv_transpose3d_atomic_kernel(
    const float* __restrict__ input,     // N, Cin, Di, Hi, Wi
    const float* __restrict__ weight,    // Cin, Cout, kD, kH, kW
    float* __restrict__ output,          // N, Cout, Do, Ho, Wo
    int N, int Cin, int Cout,
    int Di, int Hi, int Wi,
    int kD, int kH, int kW,
    int sD, int sH, int sW,
    int pD, int pH, int pW,
    int outD, int outH, int outW)
{
    int n  = blockIdx.x;
    int ic = blockIdx.y;

    int voxel = threadIdx.x;
    int num_voxels = Di * Hi * Wi;

    for (int idx = voxel; idx < num_voxels; idx += blockDim.x)
    {
        int iw = idx % Wi;
        int ih = (idx / Wi) % Hi;
        int id =  idx / (Wi * Hi);

        float in_val = input[ (((n*Cin + ic)*Di + id)*Hi + ih)*Wi + iw ];

        // Iterate kernel (flipped for conv_transpose)
        for (int kd = 0; kd < kD; ++kd)
        {
            int od = id * sD - pD + (kD - 1 - kd);
            if (od < 0 || od >= outD) continue;
            for (int kh = 0; kh < kH; ++kh)
            {
                int oh = ih * sH - pH + (kH - 1 - kh);
                if (oh < 0 || oh >= outH) continue;
                for (int kw = 0; kw < kW; ++kw)
                {
                    int ow = iw * sW - pW + (kW - 1 - kw);
                    if (ow < 0 || ow >= outW) continue;

                    for (int oc = 0; oc < Cout; ++oc)
                    {
                        int w_index = (((((ic*Cout)+oc)*kD + kd)*kH + kh)*kW + kw);
                        float w_val = weight[w_index];
                        int out_index = ((((n*Cout + oc)*outD + od)*outH + oh)*outW + ow);
                        atomicAdd(&output[out_index], in_val * w_val);
                    }
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////
// Add bias (per-channel)               output += bias[c]
////////////////////////////////////////////////////////////////
__global__ void add_bias_kernel(float* data,
                                const float* __restrict__ bias,
                                int N, int C, int D, int H, int W)
{
    CUDA_1D_KERNEL_LOOP(idx, N*C*D*H*W)
    {
        int w = idx % W;
        int tmp = idx / W;
        int h = tmp % H;
        tmp /= H;
        int d = tmp % D;
        tmp /= D;
        int c = tmp % C;
        data[idx] += bias[c];
    }
}

////////////////////////////////////////////////////////////////
// Add scalar
////////////////////////////////////////////////////////////////
__global__ void add_scalar_kernel(float* data, float scalar, int N)
{
    CUDA_1D_KERNEL_LOOP(idx, N)
    {
        data[idx] += scalar;
    }
}

////////////////////////////////////////////////////////////////
// LayerNorm across channel dimension (norm_shape = C)
////////////////////////////////////////////////////////////////
__global__ void layernorm_c_kernel(
        const float* __restrict__ input,
        const float* __restrict__ gamma,
        const float* __restrict__ beta,
        float* __restrict__ output,
        int N, int C, int D, int H, int W,
        float eps)
{
    CUDA_1D_KERNEL_LOOP(idx, N*C*D*H*W)
    {
        int w  = idx % W;
        int tmp = idx / W;
        int h  = tmp % H;
        tmp   /= H;
        int d  = tmp % D;
        tmp   /= D;
        int c  = tmp % C;
        int n  = tmp / C;

        float mean = 0.0f;
        for (int cc = 0; cc < C; ++cc)
        {
            int lin = ((((n*C)+cc)*D + d)*H + h)*W + w;
            mean += input[lin];
        }
        mean /= C;

        float var = 0.0f;
        for (int cc = 0; cc < C; ++cc)
        {
            int lin = ((((n*C)+cc)*D + d)*H + h)*W + w;
            float diff = input[lin] - mean;
            var += diff * diff;
        }
        var /= C;

        float inv_std = rsqrtf(var + eps);

        float val = input[idx];
        float normed = (val - mean) * inv_std;
        output[idx] = normed * gamma[c] + beta[c];
    }
}

////////////////////////////////////////////////////////////////
// AvgPool3d (kernel == stride)
////////////////////////////////////////////////////////////////
__global__ void avgpool3d_kernel(
        const float* __restrict__ input,
        float* __restrict__ output,
        int N, int C,
        int Di, int Hi, int Wi,
        int kD, int kH, int kW,
        int Do, int Ho, int Wo)
{
    CUDA_1D_KERNEL_LOOP(idx, N*C*Do*Ho*Wo)
    {
        int ow = idx % Wo;
        int tmp = idx / Wo;
        int oh = tmp % Ho;
        tmp   /= Ho;
        int od = tmp % Do;
        tmp   /= Do;
        int c  = tmp % C;
        int n  = tmp / C;

        int id_start = od * kD;
        int ih_start = oh * kH;
        int iw_start = ow * kW;

        float sum = 0.0f;
        for (int kd = 0; kd < kD; ++kd)
            for (int kh = 0; kh < kH; ++kh)
                for (int kw = 0; kw < kW; ++kw)
                {
                    int id = id_start + kd;
                    int ih = ih_start + kh;
                    int iw = iw_start + kw;
                    int lin = ((((n*C)+c)*Di + id)*Hi + ih)*Wi + iw;
                    sum += input[lin];
                }
        output[idx] = sum / (float)(kD*kH*kW);
    }
}

////////////////////////////////////////////////////////////////
// GELU (approx)
////////////////////////////////////////////////////////////////
__global__ void gelu_kernel(float* data, int N)
{
    CUDA_1D_KERNEL_LOOP(idx, N)
    {
        float x = data[idx];
        float y = 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
        data[idx] = y;
    }
}

////////////////////////////////////////////////////////////////
// Host exposed wrappers
////////////////////////////////////////////////////////////////
void conv_transpose3d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor output,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding)
{
    const int N    = input.size(0);
    const int Cin  = input.size(1);
    const int Di   = input.size(2);
    const int Hi   = input.size(3);
    const int Wi   = input.size(4);

    const int Cout = weight.size(1);
    const int kD   = weight.size(2);
    const int kH   = weight.size(3);
    const int kW   = weight.size(4);

    const int sD = stride[0], sH = stride[1], sW = stride[2];
    const int pD = padding[0], pH = padding[1], pW = padding[2];

    const int outD = output.size(2);
    const int outH = output.size(3);
    const int outW = output.size(4);

    output.zero_();

    dim3 grid(N, Cin);
    const int threads = 128;
    conv_transpose3d_atomic_kernel<<<grid, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        N, Cin, Cout,
        Di, Hi, Wi,
        kD, kH, kW,
        sD, sH, sW,
        pD, pH, pW,
        outD, outH, outW
    );

    add_bias_kernel<<<GET_BLOCKS(N*Cout*outD*outH*outW), 256>>>(
        output.data_ptr<float>(),
        bias.data_ptr<float>(),
        N, Cout, outD, outH, outW
    );
}

void add_scalar_forward(torch::Tensor tensor, float scalar)
{
    int N = tensor.numel();
    add_scalar_kernel<<<GET_BLOCKS(N), 256>>>(tensor.data_ptr<float>(), scalar, N);
}

void layernorm_forward(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, torch::Tensor output, float eps)
{
    int N=input.size(0), C=input.size(1), D=input.size(2), H=input.size(3), W=input.size(4);
    int total = N*C*D*H*W;
    layernorm_c_kernel<<<GET_BLOCKS(total), 256>>>(
        input.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        N,C,D,H,W,eps
    );
}

void avgpool3d_forward(torch::Tensor input, torch::Tensor output, std::vector<int64_t> ksize)
{
    int kD=ksize[0], kH=ksize[1], kW=ksize[2];
    int N=input.size(0), C=input.size(1), Di=input.size(2), Hi=input.size(3), Wi=input.size(4);
    int Do=output.size(2), Ho=output.size(3), Wo=output.size(4);
    int total = N*C*Do*Ho*Wo;
    avgpool3d_kernel<<<GET_BLOCKS(total), 256>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N,C,Di,Hi,Wi,
        kD,kH,kW,
        Do,Ho,Wo
    );
}

void gelu_forward(torch::Tensor tensor)
{
    int N = tensor.numel();
    gelu_kernel<<<GET_BLOCKS(N), 256>>>(tensor.data_ptr<float>(), N);
}
"""

cpp_src = """
void conv_transpose3d_forward(torch::Tensor input,
                              torch::Tensor weight,
                              torch::Tensor bias,
                              torch::Tensor output,
                              std::vector<int64_t> stride,
                              std::vector<int64_t> padding);
void add_scalar_forward(torch::Tensor tensor, float scalar);
void layernorm_forward(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, torch::Tensor output, float eps);
void avgpool3d_forward(torch::Tensor input, torch::Tensor output, std::vector<int64_t> ksize);
void gelu_forward(torch::Tensor tensor);
"""

# Compile
cuda_fused = load_inline(
    name="cuda_fused_ops_fixed",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=[
        "conv_transpose3d_forward",
        "add_scalar_forward",
        "layernorm_forward",
        "avgpool3d_forward",
        "gelu_forward",
    ],
    verbose=False,
)

# ------------------------------------------------------------------
# Optimised Model
# ------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Hand-written CUDA kernel replacement model.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, output_padding, sum_weight, norm_shape,
                 pool_kernel_size):
        super(ModelNew, self).__init__()
        kD, kH, kW = kernel_size
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, kD, kH, kW))
        self.bias   = nn.Parameter(torch.zeros(out_channels))

        self.register_buffer("sum_weight", torch.tensor(float(sum_weight)))

        self.gamma = nn.Parameter(torch.ones(norm_shape))
        self.beta  = nn.Parameter(torch.zeros(norm_shape))

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.pool_kernel_size = pool_kernel_size
        self.eps = 1e-5

        self.kernels = cuda_fused

    def forward(self, x):
        N, _, Di, Hi, Wi = x.shape
        outD = (Di - 1) * self.stride[0] - 2 * self.padding[0] + \
               self.kernel_size[0] + self.output_padding[0]
        outH = (Hi - 1) * self.stride[1] - 2 * self.padding[1] + \
               self.kernel_size[1] + self.output_padding[1]
        outW = (Wi - 1) * self.stride[2] - 2 * self.padding[2] + \
               self.kernel_size[2] + self.output_padding[2]

        device = x.device
        out = torch.zeros((N, self.weight.shape[1], outD, outH, outW),
                          device=device, dtype=x.dtype)

        self.kernels.conv_transpose3d_forward(
            x.contiguous(), self.weight, self.bias,
            out,
            list(self.stride),
            list(self.padding)
        )

        self.kernels.add_scalar_forward(out, float(self.sum_weight))

        ln_out = torch.empty_like(out)
        self.kernels.layernorm_forward(
            out.contiguous(),
            self.gamma,
            self.beta,
            ln_out,
            self.eps
        )

        N, C, D1, H1, W1 = ln_out.shape
        kD, kH, kW = self.pool_kernel_size
        Do = D1 // kD
        Ho = H1 // kH
        Wo = W1 // kW
        pool_out = torch.empty((N, C, Do, Ho, Wo), device=device, dtype=ln_out.dtype)
        self.kernels.avgpool3d_forward(
            ln_out.contiguous(),
            pool_out,
            list(self.pool_kernel_size)
        )

        self.kernels.gelu_forward(pool_out)

        return pool_out
