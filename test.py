import argparse
import time
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------
# CUDA kernel source (kernels + host wrappers; NO pybind module here)
# —— 直接使用你提供的代码 —— 
# ---------------------------------------------------------------------
source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT32(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")

__global__ void conv2d_forward_kernel_fused(
        const float* __restrict__ input,      // [N, Ci, H, W]
        const float* __restrict__ weight,     // [Co, Ci, K, K]
        const float* __restrict__ bias,       // [Co] or nullptr
        float* __restrict__ output,           // [N, Co, Ho, Wo]
        int N, int Ci, int H, int W,
        int Co, int K, int Ho, int Wo)
{
    int nc = blockIdx.z;          // fused (n, co)
    int n  = nc / Co;
    int co = nc % Co;

    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    if (h_out >= Ho || w_out >= Wo) return;

    const int in_hw_stride    = W;
    const int in_ch_stride    = H * W;
    const int in_batch_stride = Ci * H * W;

    const int wt_hw_stride    = K;
    const int wt_ch_stride    = K * K;
    const int wt_out_stride   = Ci * K * K;

    float acc = bias ? bias[co] : 0.f;

    for (int ci = 0; ci < Ci; ++ci) {
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                int h_in = h_out + kh;
                int w_in = w_out + kw;

                float in_val = input[n * in_batch_stride +
                                     ci * in_ch_stride +
                                     h_in * in_hw_stride +
                                     w_in];

                float wt_val = weight[co * wt_out_stride +
                                      ci * wt_ch_stride +
                                      kh * wt_hw_stride +
                                      kw];

                acc += in_val * wt_val;
            }
        }
    }

    output[n * (Co * Ho * Wo) +
           co * (Ho * Wo) +
           h_out * Wo +
           w_out] = acc;
}

__global__ void conv2d_forward_kernel_batch(
        const float* __restrict__ input,      // [N, Ci, H, W]
        const float* __restrict__ weight,     // [Co, Ci, K, K]
        const float* __restrict__ bias,       // [Co] or nullptr
        float* __restrict__ output,           // [N, Co, Ho, Wo]
        int N, int Ci, int H, int W,
        int Co, int K, int Ho, int Wo)
{
    int n = blockIdx.z;   // batch id

    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    if (h_out >= Ho || w_out >= Wo) return;

    const int in_hw_stride    = W;
    const int in_ch_stride    = H * W;
    const int in_batch_stride = Ci * H * W;

    const int wt_hw_stride    = K;
    const int wt_ch_stride    = K * K;
    const int wt_out_stride   = Ci * K * K;

    for (int co = 0; co < Co; ++co) {
        float acc = bias ? bias[co] : 0.f;

        for (int ci = 0; ci < Ci; ++ci) {
            for (int kh = 0; kh < K; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
                    int h_in = h_out + kh;
                    int w_in = w_out + kw;

                    float in_val = input[n * in_batch_stride +
                                         ci * in_ch_stride +
                                         h_in * in_hw_stride +
                                         w_in];

                    float wt_val = weight[co * wt_out_stride +
                                          ci * wt_ch_stride +
                                          kh * wt_hw_stride +
                                          kw];

                    acc += in_val * wt_val;
                }
            }
        }

        output[n * (Co * Ho * Wo) +
               co * (Ho * Wo) +
               h_out * Wo +
               w_out] = acc;
    }
}

template<int BLOCK_SIZE>
__global__ void calc_mean_invstd_kernel(
        const float* __restrict__ x,      // [N, C, H, W] flattened to NC*HW
        float* __restrict__ mean,         // [N*C]
        float* __restrict__ invstd,       // [N*C]
        int H, int W, int total_nc, float eps)
{
    int nc = blockIdx.x;

    while (nc < total_nc) {
        int S  = H * W;

        __shared__ float shm[2 * BLOCK_SIZE];

        float local_sum   = 0.f;
        float local_sumsq = 0.f;

        for (int idx = threadIdx.x; idx < S; idx += BLOCK_SIZE) {
            float val = x[nc * S + idx];
            local_sum   += val;
            local_sumsq += val * val;
        }

        shm[threadIdx.x]              = local_sum;
        shm[BLOCK_SIZE + threadIdx.x] = local_sumsq;
        __syncthreads();

        for (int stride = BLOCK_SIZE >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                shm[threadIdx.x]              += shm[threadIdx.x + stride];
                shm[BLOCK_SIZE + threadIdx.x] += shm[BLOCK_SIZE + threadIdx.x + stride];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            float sum   = shm[0];
            float sumsq = shm[BLOCK_SIZE];
            float mu    = sum / S;
            float var   = sumsq / S - mu * mu;
            mean[nc]    = mu;
            invstd[nc]  = rsqrtf(var + eps);
        }
        __syncthreads();
        nc += gridDim.x;
    }
}

__global__ void norm_div_kernel(
        const float* __restrict__ x,          // [N, C, H, W] flattened
        const float* __restrict__ mean,       // [N*C]
        const float* __restrict__ invstd,     // [N*C]
        float* __restrict__ y,
        int S, int total_nc, float divide_by)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int linear = idx; linear < total_nc * S; linear += stride) {
        int nc  = linear / S;
        float val = x[linear];
        float mu  = mean[nc];
        float inv = invstd[nc];
        y[linear] = ((val - mu) * inv) / divide_by;
    }
}

torch::Tensor conv2d_forward_cuda(torch::Tensor input,
                                  torch::Tensor weight,
                                  torch::Tensor bias) {
    CHECK_CUDA(input);   CHECK_CONTIGUOUS(input);   CHECK_FLOAT32(input);
    CHECK_CUDA(weight);  CHECK_CONTIGUOUS(weight);  CHECK_FLOAT32(weight);
    if (bias.defined()) { CHECK_CUDA(bias); CHECK_CONTIGUOUS(bias); CHECK_FLOAT32(bias); }

    int N  = input.size(0);
    int Ci = input.size(1);
    int H  = input.size(2);
    int W  = input.size(3);
    int Co = weight.size(0);
    int K  = weight.size(2);

    int Ho = H - K + 1;
    int Wo = W - K + 1;

    auto output = torch::empty({N, Co, Ho, Wo}, input.options());

    dim3 block(16, 16);
    dim3 grid((Wo + block.x - 1) / block.x,
              (Ho + block.y - 1) / block.y,
              1);

    const int64_t max_grid_z = 65535;

    if (static_cast<int64_t>(N) * Co <= max_grid_z) {
        grid.z = N * Co;
        conv2d_forward_kernel_fused<<<grid, block>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.defined() ? bias.data_ptr<float>() : nullptr,
            output.data_ptr<float>(),
            N, Ci, H, W, Co, K, Ho, Wo);
    } else {
        grid.z = N;
        conv2d_forward_kernel_batch<<<grid, block>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.defined() ? bias.data_ptr<float>() : nullptr,
            output.data_ptr<float>(),
            N, Ci, H, W, Co, K, Ho, Wo);
    }

    return output;
}

torch::Tensor instance_norm_divide_cuda(torch::Tensor input,
                                        double divide_by,
                                        double eps = 1e-5) {
    CHECK_CUDA(input); CHECK_CONTIGUOUS(input); CHECK_FLOAT32(input);

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int S = H * W;
    int total_nc = N * C;

    auto output = torch::empty_like(input);
    auto mean   = torch::empty({total_nc}, input.options());
    auto invstd = torch::empty({total_nc}, input.options());

    constexpr int BLOCK_SIZE = 256;
    int max_blocks_stat = 65535;
    int grid_stat = total_nc < max_blocks_stat ? total_nc : max_blocks_stat;

    calc_mean_invstd_kernel<BLOCK_SIZE><<<grid_stat, BLOCK_SIZE>>>(
        input.data_ptr<float>(),
        mean.data_ptr<float>(),
        invstd.data_ptr<float>(),
        H, W, total_nc, static_cast<float>(eps));

    int threads = 256;
    int64_t total_elems = static_cast<int64_t>(total_nc) * S;
    int64_t blocks_long = (total_elems + threads - 1) / threads;
    int grid_norm = static_cast<int>(std::min<int64_t>(blocks_long, 65535));

    norm_div_kernel<<<grid_norm, threads>>>(
        input.data_ptr<float>(),
        mean.data_ptr<float>(),
        invstd.data_ptr<float>(),
        output.data_ptr<float>(),
        S, total_nc, static_cast<float>(divide_by));

    return output;
}
"""

# ---------------------------------------------------------------------
# C++ prototypes（无 pybind 模块定义）
# ---------------------------------------------------------------------
cpp_src = r"""
#include <torch/extension.h>

torch::Tensor conv2d_forward_cuda(torch::Tensor input,
                                  torch::Tensor weight,
                                  torch::Tensor bias);

torch::Tensor instance_norm_divide_cuda(torch::Tensor input,
                                        double divide_by,
                                        double eps = 1e-5);
"""

# ---------------------------------------------------------------------
# Compile & load
# ---------------------------------------------------------------------
kernels = load_inline(
    name="fused_conv_instnorm_v1_fixed",
    cpp_sources=[cpp_src],
    cuda_sources=[source],
    functions=[
        "conv2d_forward_cuda",
        "instance_norm_divide_cuda",
    ],
    extra_cuda_cflags=[],
    verbose=False,
)

# ---------------------------------------------------------------------
# PyTorch Module wrapper（与你提供的一致）
# ---------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Conv2d (stride=1, padding=0) -> InstanceNorm (affine=False) -> divide_by
    使用自定义 CUDA kernel 实现。
    """
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super().__init__()
        weight = torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        nn.init.kaiming_uniform_(weight, a=5 ** 0.5)
        bias = torch.empty(out_channels)
        fan_in = in_channels * kernel_size * kernel_size
        bound = 1 / fan_in ** 0.5
        nn.init.uniform_(bias, -bound, bound)

        self.weight = nn.Parameter(weight)
        self.bias   = nn.Parameter(bias)
        self.divide_by = float(divide_by)

    def forward(self, x):
        x = kernels.conv2d_forward_cuda(
            x.contiguous(),
            self.weight.contiguous(),
            self.bias.contiguous()
        )
        x = kernels.instance_norm_divide_cuda(
            x.contiguous(),
            self.divide_by
        )
        return x

# ---------------------------------------------------------------------
# 参考实现（PyTorch 原生模块）
# ---------------------------------------------------------------------
class RefModel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divide_by, eps=1e-5):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=1, padding=0, bias=True)
        # 与自定义 kernel 对齐：每个样本/通道统计，affine=False，track_running_stats=False
        self.inorm = nn.InstanceNorm2d(out_channels, affine=False, track_running_stats=False, eps=eps)
        self.divide_by = float(divide_by)

    def forward(self, x):
        y = self.conv(x)
        y = self.inorm(y)
        return y / self.divide_by

# ---------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------
@torch.inference_mode()
def check_correctness(model_cuda, model_ref, x, atol=1e-4, rtol=1e-4):
    y_cuda = model_cuda(x)
    y_ref  = model_ref(x)
    ok = torch.allclose(y_cuda, y_ref, atol=atol, rtol=rtol)
    max_abs = (y_cuda - y_ref).abs().max().item()
    mean_abs = (y_cuda - y_ref).abs().mean().item()
    return ok, max_abs, mean_abs, y_cuda, y_ref

@torch.inference_mode()
def bench(fn, x, warmup=10, repeat=50):
    # 预热
    for _ in range(warmup):
        _ = fn(x)
    torch.cuda.synchronize()
    # 正式计时
    t0 = time.perf_counter()
    for _ in range(repeat):
        _ = fn(x)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / repeat  # ms/iter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=8)
    parser.add_argument("--Ci", type=int, default=16)
    parser.add_argument("--Co", type=int, default=32)
    parser.add_argument("--H", type=int, default=64)
    parser.add_argument("--W", type=int, default=64)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--divide_by", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA device is required."
    device = torch.device("cuda")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # 形状检查（valid conv，无 padding）
    assert args.H >= args.K and args.W >= args.K, "H/W must be >= K for valid conv (no padding)."

    # 构造模型与输入
    model_cuda = ModelNew(args.Ci, args.Co, args.K, args.divide_by).to(device)
    model_ref  = RefModel(args.Ci, args.Co, args.K, args.divide_by).to(device)

    # 将参考模型的权重/偏置与自定义模型保持一致
    with torch.no_grad():
        model_ref.conv.weight.copy_(model_cuda.weight.detach())
        model_ref.conv.bias.copy_(model_cuda.bias.detach())

    x = torch.randn(args.N, args.Ci, args.H, args.W, device=device, dtype=torch.float32)

    # 数值正确性
    ok, max_abs, mean_abs, y_cuda, y_ref = check_correctness(
        model_cuda, model_ref, x, atol=args.atol, rtol=args.rtol
    )
    print(f"[Correctness] allclose={ok} (atol={args.atol}, rtol={args.rtol})")
    print(f"[Correctness] max|diff| = {max_abs:.6e}, mean|diff| = {mean_abs:.6e}")

    # 基准测试
    ms_ref  = bench(model_ref, x, warmup=args.warmup, repeat=args.repeat)
    ms_cuda = bench(model_cuda, x, warmup=args.warmup, repeat=args.repeat)
    speedup = ms_ref / ms_cuda if ms_cuda > 0 else float("inf")

    print("\n[Benchmark]")
    print(f"  Ref (PyTorch) : {ms_ref:.3f} ms / iter")
    print(f"  CUDA kernels  : {ms_cuda:.3f} ms / iter")
    print(f"  Speedup       : {speedup:.2f}x")

    # 额外打印输出形状以核对
    print(f"\nOutput shape: {tuple(y_cuda.shape)}  |  Expected: (N, Co, H-K+1, W-K+1) = ({args.N}, {args.Co}, {args.H-args.K+1}, {args.W-args.K+1})")

if __name__ == "__main__":
    main()
