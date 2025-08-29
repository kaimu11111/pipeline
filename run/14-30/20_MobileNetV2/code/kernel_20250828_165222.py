# 1. Imports ────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# 2. source – CUDA kernels + host wrappers ──────────────────────────────
source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// ---------------- CUDA kernels ----------------
template <typename scalar_t>
__global__ void relu6_forward_kernel(const scalar_t* __restrict__ x,
                                     scalar_t* __restrict__ y,
                                     int64_t numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int64_t i = idx; i < numel; i += blockDim.x * gridDim.x) {
        scalar_t v = x[i];
        v = v > scalar_t(0) ? v : scalar_t(0);
        v = v < scalar_t(6) ? v : scalar_t(6);
        y[i] = v;
    }
}

template <typename scalar_t>
__global__ void relu6_backward_kernel(const scalar_t* __restrict__ grad_out,
                                      const scalar_t* __restrict__ x,
                                      scalar_t* __restrict__ grad_in,
                                      int64_t numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int64_t i = idx; i < numel; i += blockDim.x * gridDim.x) {
        scalar_t v = x[i];
        grad_in[i] = (v > scalar_t(0) && v < scalar_t(6))
                         ? grad_out[i]
                         : scalar_t(0);
    }
}

// -------------- Host-side CUDA wrappers ---------------
torch::Tensor relu6_forward_cuda(torch::Tensor x) {
    auto y          = torch::empty_like(x);
    const int64_t N = x.numel();
    const int  thr  = 256;
    const int  blk  = (N + thr - 1) / thr;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        x.scalar_type(), "relu6_forward_cuda", ([&] {
            relu6_forward_kernel<scalar_t><<<blk, thr>>>(
                x.data_ptr<scalar_t>(),
                y.data_ptr<scalar_t>(),
                N);
        }));
    return y;
}

torch::Tensor relu6_backward_cuda(torch::Tensor grad_out,
                                  torch::Tensor x) {
    auto grad_in    = torch::empty_like(x);
    const int64_t N = x.numel();
    const int  thr  = 256;
    const int  blk  = (N + thr - 1) / thr;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        x.scalar_type(), "relu6_backward_cuda", ([&] {
            relu6_backward_kernel<scalar_t><<<blk, thr>>>(
                grad_out.data_ptr<scalar_t>(),
                x.data_ptr<scalar_t>(),
                grad_in.data_ptr<scalar_t>(),
                N);
        }));
    return grad_in;
}
"""

# 3. cpp_src – exposed prototypes ───────────────────────────────────────
cpp_src = r"""
torch::Tensor relu6_forward_cuda(torch::Tensor x);
torch::Tensor relu6_backward_cuda(torch::Tensor grad_out, torch::Tensor x);
"""

# 4. load_inline call ───────────────────────────────────────────────────
relu6_ext = load_inline(
    name="fast_relu6_ext",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["relu6_forward_cuda", "relu6_backward_cuda"],
    verbose=False,
)

# Helper: autograd-aware Python wrapper
class _FastReLU6Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x_ = x.contiguous()
        y  = relu6_ext.relu6_forward_cuda(x_)
        ctx.save_for_backward(x_)
        return y

    @staticmethod
    def backward(ctx, gy):
        (x_,) = ctx.saved_tensors
        return relu6_ext.relu6_backward_cuda(gy.contiguous(), x_)

class FastReLU6(nn.Module):
    def forward(self, x):
        return _FastReLU6Func.apply(x)

# 5. ModelNew ───────────────────────────────────────────────────────────
class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        def _make_divisible(v, d, min_v=None):
            if min_v is None:
                min_v = d
            new_v = max(min_v, int(v + d / 2) // d * d)
            if new_v < 0.9 * v:
                new_v += d
            return new_v

        def _inv_res_block(inp, oup, stride, exp):
            hid = int(inp * exp)
            layers = []
            if exp != 1:
                layers += [nn.Conv2d(inp, hid, 1, bias=False),
                           nn.BatchNorm2d(hid),
                           FastReLU6()]
            layers += [nn.Conv2d(hid, hid, 3, stride, 1,
                                 groups=hid, bias=False),
                       nn.BatchNorm2d(hid),
                       FastReLU6(),
                       nn.Conv2d(hid, oup, 1, bias=False),
                       nn.BatchNorm2d(oup)]
            return nn.Sequential(*layers)

        input_c   = 32
        last_c    = 1280
        settings  = [
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        feats = [nn.Conv2d(3, input_c, 3, 2, 1, bias=False),
                 nn.BatchNorm2d(input_c),
                 FastReLU6()]

        for t, c, n, s in settings:
            out_c = _make_divisible(c, 8)
            for i in range(n):
                stride = s if i == 0 else 1
                feats.append(_inv_res_block(input_c, out_c, stride, t))
                input_c = out_c

        feats += [nn.Conv2d(input_c, last_c, 1, bias=False),
                  nn.BatchNorm2d(last_c),
                  FastReLU6(),
                  nn.AdaptiveAvgPool2d(1)]

        self.features = nn.Sequential(*feats)
        self.classifier = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(last_c, num_classes)
        )

        # Weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)
