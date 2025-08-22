import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------------
# 1. CUDA source code ------------------------------------------------
# ------------------------------------------------------------------
source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// ------------------------------------------------------------
// 3-D convolution (stride=1, padding=0, dilation=1)
// ------------------------------------------------------------
template<typename scalar_t>
__global__ void conv3d_kernel(
        const scalar_t* __restrict__ in,
        const scalar_t* __restrict__ weight,
        scalar_t* __restrict__ out,
        const int N, const int C_in,
        const int D, const int H, const int W,
        const int Kd, const int Kh, const int Kw,
        const int C_out) {

    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int d_out = blockIdx.z * blockDim.z + threadIdx.z;
    if (w_out >= (W - Kw + 1) || h_out >= (H - Kh + 1) || d_out >= (D - Kd + 1))
        return;

    for (int n = 0; n < N; ++n) {
        for (int co = 0; co < C_out; ++co) {
            scalar_t sum = 0;
            for (int ci = 0; ci < C_in; ++ci) {
                for (int kd = 0; kd < Kd; ++kd) {
                    for (int kh = 0; kh < Kh; ++kh) {
                        for (int kw = 0; kw < Kw; ++kw) {
                            int d_in = d_out + kd;
                            int h_in = h_out + kh;
                            int w_in = w_out + kw;

                            size_t input_idx  = (((n * C_in + ci) * D + d_in) * H + h_in) * W + w_in;
                            size_t weight_idx = (((co * C_in + ci) * Kd + kd) * Kh + kh) * Kw + kw;
                            sum += in[input_idx] * weight[weight_idx];
                        }
                    }
                }
            }
            size_t out_idx = (((n * C_out + co) * (D - Kd + 1) + d_out) * (H - Kh + 1) + h_out) * (W - Kw + 1) + w_out;
            out[out_idx] = sum;
        }
    }
}

torch::Tensor conv3d_forward_cuda(torch::Tensor input, torch::Tensor weight) {
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Only float32 supported");

    const int N  = input.size(0);
    const int C  = input.size(1);
    const int D  = input.size(2);
    const int H  = input.size(3);
    const int W  = input.size(4);

    const int Cout = weight.size(0);
    const int Kd = weight.size(2);
    const int Kh = weight.size(3);
    const int Kw = weight.size(4);

    const int Dout = D - Kd + 1;
    const int Hout = H - Kh + 1;
    const int Wout = W - Kw + 1;

    auto out = torch::empty({N, Cout, Dout, Hout, Wout}, input.options());

    dim3 block(4,4,4);
    dim3 grid((Wout + block.x -1)/block.x,
              (Hout + block.y -1)/block.y,
              (Dout + block.z -1)/block.z);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv3d_forward_cuda", ([&]{
        conv3d_kernel<scalar_t><<<grid, block>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            N, C, D, H, W,
            Kd, Kh, Kw,
            Cout);
    }));
    cudaDeviceSynchronize();
    return out;
}

// ------------------------------------------------------------
// scale by constant (division, to match reference behaviour)
// ------------------------------------------------------------
template<typename scalar_t>
__global__ void scale_const_kernel(const scalar_t* in, scalar_t* out, const scalar_t factor, int64_t numel){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < numel){
        out[idx] = in[idx] / factor;
    }
}

torch::Tensor scale_const_cuda(torch::Tensor input, double factor){
    CHECK_INPUT(input);
    auto out = torch::empty_like(input);
    int64_t numel = input.numel();
    const int block = 256;
    const int grid  = (numel + block - 1) / block;
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "scale_const_cuda", ([&]{
        scale_const_kernel<scalar_t><<<grid, block>>>(
            input.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), (scalar_t)factor, numel);
    }));
    cudaDeviceSynchronize();
    return out;
}

// ------------------------------------------------------------
// 3-D MaxPool (window & stride specified)
// ------------------------------------------------------------
template<typename scalar_t>
__global__ void maxpool3d_kernel(
        const scalar_t* __restrict__ in,
        scalar_t* __restrict__ out,
        int N, int C,
        int D, int H, int W,
        int Kd, int Kh, int Kw,
        int Sd, int Sh, int Sw,
        int Dout, int Hout, int Wout){

    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int d_out = blockIdx.z * blockDim.z + threadIdx.z;
    if (w_out >= Wout || h_out >= Hout || d_out >= Dout) return;

    for(int n=0;n<N;++n){
        for(int c=0;c<C;++c){
            scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
            for(int kd=0;kd<Kd;++kd){
                for(int kh=0;kh<Kh;++kh){
                    for(int kw=0;kw<Kw;++kw){
                        int d_in = d_out*Sd + kd;
                        int h_in = h_out*Sh + kh;
                        int w_in = w_out*Sw + kw;
                        size_t idx = (((n*C + c)*D + d_in)*H + h_in)*W + w_in;
                        scalar_t v = in[idx];
                        if(v > max_val) max_val = v;
                    }
                }
            }
            size_t oidx = (((n*C + c)*Dout + d_out)*Hout + h_out)*Wout + w_out;
            out[oidx] = max_val;
        }
    }
}

torch::Tensor maxpool3d_cuda(torch::Tensor input, std::vector<int64_t> kernel_size){
    CHECK_INPUT(input);
    TORCH_CHECK(kernel_size.size()==3, "kernel_size must be 3");

    int Kd = kernel_size[0], Kh = kernel_size[1], Kw = kernel_size[2];
    int Sd = Kd, Sh = Kh, Sw = Kw;

    const int N = input.size(0);
    const int C = input.size(1);
    const int D = input.size(2);
    const int H = input.size(3);
    const int W = input.size(4);

    int Dout = (D - Kd) / Sd + 1;
    int Hout = (H - Kh) / Sh + 1;
    int Wout = (W - Kw) / Sw + 1;

    auto out = torch::empty({N,C,Dout,Hout,Wout}, input.options());

    dim3 block(4,4,4);
    dim3 grid((Wout + block.x -1)/block.x,
              (Hout + block.y -1)/block.y,
              (Dout + block.z -1)/block.z);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "maxpool3d_cuda", ([&]{
        maxpool3d_kernel<scalar_t><<<grid, block>>>(
            input.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            N, C, D, H, W,
            Kd, Kh, Kw,
            Sd, Sh, Sw,
            Dout, Hout, Wout);
    }));
    cudaDeviceSynchronize();
    return out;
}

// ------------------------------------------------------------
// Global Average Pooling to (1,1,1)
// ------------------------------------------------------------
template<typename scalar_t>
__global__ void gap3d_kernel(const scalar_t* in, scalar_t* out,
                             int N, int C, int D, int H, int W, int spatial){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N*C) return;
    int n = idx / C;
    int c = idx % C;
    scalar_t sum = 0;
    for(int d=0; d<D; ++d)
        for(int h=0; h<H; ++h)
            for(int w=0; w<W; ++w){
                size_t offset = (((n*C + c)*D + d)*H + h)*W + w;
                sum += in[offset];
            }
    out[idx] = sum / static_cast<scalar_t>(spatial);
}

torch::Tensor global_avg_pool3d_cuda(torch::Tensor input){
    CHECK_INPUT(input);
    int N = input.size(0);
    int C = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);
    int spatial = D*H*W;
    auto out = torch::empty({N,C,1,1,1}, input.options());
    int threads = 256;
    int blocks  = (N*C + threads -1)/threads;
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "global_avg_pool3d_cuda", ([&]{
        gap3d_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),
            N,C,D,H,W,spatial);
    }));
    cudaDeviceSynchronize();
    return out;
}

// ------------------------------------------------------------
// Add bias (shape Cx1x1x1) to tensor (N,C,1,1,1)
// ------------------------------------------------------------
template<typename scalar_t>
__global__ void add_bias_kernel(const scalar_t* in, const scalar_t* bias, scalar_t* out, int N, int C){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N*C) return;
    int c = idx % C;
    out[idx] = in[idx] + bias[c];
}

torch::Tensor add_bias_cuda(torch::Tensor input, torch::Tensor bias){
    CHECK_INPUT(input);
    CHECK_INPUT(bias);
    int N = input.size(0);
    int C = input.size(1);
    auto out = torch::empty_like(input);
    int threads = 256;
    int blocks = (N*C + threads -1)/threads;
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "add_bias_cuda", ([&]{
        add_bias_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            N,C);
    }));
    cudaDeviceSynchronize();
    return out;
}

// ------------------------------------------------------------
// Sum along channel dimension (dim=1) of shape (N,C,1,1,1)
// ------------------------------------------------------------
__global__ void sum_dim1_kernel_float(const float* in, float* out, int N, int C){
    int n = blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ float shared_data[];
    float val = 0.f;
    for(int c = tid; c < C; c += blockDim.x){
        val += in[n*C + c];
    }
    shared_data[tid] = val;
    __syncthreads();
    for(int s = blockDim.x / 2; s > 0; s >>= 1){
        if(tid < s){
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }
    if(tid == 0){
        out[n] = shared_data[0];
    }
}

torch::Tensor sum_dim1_cuda(torch::Tensor input){
    CHECK_INPUT(input);
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "sum_dim1_cuda supports only float32");
    int N = input.size(0);
    int C = input.size(1);
    auto out = torch::empty({N}, input.options());
    int threads = 256;
    int shared = threads * sizeof(float);
    sum_dim1_kernel_float<<<N, threads, shared>>>(
        input.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C);
    cudaDeviceSynchronize();
    return out;
}
"""

# ------------------------------------------------------------------
# 2. C++ prototypes -------------------------------------------------
# ------------------------------------------------------------------
cpp_src = """
torch::Tensor conv3d_forward_cuda(torch::Tensor input, torch::Tensor weight);
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
    name="custom_cuda_kernels_fixed",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=[
        "conv3d_forward_cuda",
        "scale_const_cuda",
        "maxpool3d_cuda",
        "global_avg_pool3d_cuda",
        "add_bias_cuda",
        "sum_dim1_cuda",
    ],
    verbose=False,
)

# ------------------------------------------------------------------
# 4. Python wrapper Model ------------------------------------------
# ------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Custom CUDA-optimised model replacing all PyTorch ops with hand-written kernels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, factor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        kd, kh, kw = kernel_size
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kd, kh, kw, device='cuda', dtype=torch.float32))
        self.bias   = nn.Parameter(torch.randn(*bias_shape, device='cuda', dtype=torch.float32))
        self.factor = float(factor)
        self.pool_size = pool_size
        self.sum_dim = sum_dim  # assumed 1

    def forward(self, x):
        x = x.contiguous().float()
        x = kernels.conv3d_forward_cuda(x, self.weight)
        x = kernels.scale_const_cuda(x, self.factor)
        x = kernels.maxpool3d_cuda(x, list(self.pool_size))
        x = kernels.global_avg_pool3d_cuda(x)
        x = kernels.add_bias_cuda(x, self.bias)
        if self.sum_dim == 1:
            x = kernels.sum_dim1_cuda(x)
        else:
            raise NotImplementedError("Only sum_dim==1 supported in custom kernel")
        return x
