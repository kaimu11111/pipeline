import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# ------------- CUDA: Fused BatchNorm2d + ReLU kernel -----------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_bn_relu_kernel(const float* __restrict__ input,
                                     const float* __restrict__ weight,
                                     const float* __restrict__ bias,
                                     const float* __restrict__ running_mean,
                                     const float* __restrict__ running_var,
                                     float* __restrict__ output,
                                     int C,
                                     int spatial_size,
                                     float eps,
                                     int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int c = (idx / spatial_size) % C;
    float inv_std = rsqrtf(running_var[c] + eps);
    float val = (input[idx] - running_mean[c]) * inv_std * weight[c] + bias[c];
    output[idx] = fmaxf(val, 0.0f);  // ReLU
}

torch::Tensor fused_bn_relu_cuda(torch::Tensor input,
                                 torch::Tensor weight,
                                 torch::Tensor bias,
                                 torch::Tensor running_mean,
                                 torch::Tensor running_var,
                                 double eps) {
    // Expect contiguous float tensors on CUDA
    input        = input.contiguous();
    weight       = weight.contiguous();
    bias         = bias.contiguous();
    running_mean = running_mean.contiguous();
    running_var  = running_var.contiguous();

    auto output = torch::empty_like(input);

    const int C              = input.size(1);
    const int spatial_size   = input.size(2) * input.size(3);
    const int total_elements = input.numel();

    constexpr int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    fused_bn_relu_kernel<<<blocks, threads>>>(input.data_ptr<float>(),
                                              weight.data_ptr<float>(),
                                              bias.data_ptr<float>(),
                                              running_mean.data_ptr<float>(),
                                              running_var.data_ptr<float>(),
                                              output.data_ptr<float>(),
                                              C,
                                              spatial_size,
                                              static_cast<float>(eps),
                                              total_elements);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel failed : %s\n", cudaGetErrorString(err));
    }
    return output;
}
"""

cpp_decls = """
torch::Tensor fused_bn_relu_cuda(torch::Tensor input,
                                 torch::Tensor weight,
                                 torch::Tensor bias,
                                 torch::Tensor running_mean,
                                 torch::Tensor running_var,
                                 double eps);
"""

fused_bn_relu = load_inline(
    name="fused_bn_relu",
    cpp_sources=cpp_decls,
    cuda_sources=cuda_source,
    functions=["fused_bn_relu_cuda"],
    verbose=False,
)

# ------------- Python wrapper module ---------------------------------
class FusedBatchNormReLU2d(nn.Module):
    """
    Replaces a BatchNorm2d + ReLU pair with a single fused CUDA kernel
    (inference mode). Falls back to standard ops during training or on CPU.
    """
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Training mode or CPU tensors fall back to native implementation
        if self.training or not x.is_cuda:
            return F.relu(self.bn(x), inplace=True)
        # Inference on CUDA uses fused kernel
        return fused_bn_relu.fused_bn_relu_cuda(
            x,
            self.bn.weight,
            self.bn.bias,
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.eps,
        )

# ------------- Building blocks using fused BN+ReLU -------------------
class DenseBlock(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            in_feat = num_input_features + i * growth_rate
            layers.append(self._make_layer(in_feat, growth_rate))
        self.layers = nn.ModuleList(layers)

    def _make_layer(self, in_features: int, growth_rate: int):
        return nn.Sequential(
            FusedBatchNormReLU2d(in_features),
            nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.Dropout(0.0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for layer in self.layers:
            new_feature = layer(x)
            features.append(new_feature)
            x = torch.cat(features, 1)  # Concatenate along the channel dimension
        return x

class TransitionLayer(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        super(TransitionLayer, self).__init__()
        self.transition = nn.Sequential(
            FusedBatchNormReLU2d(num_input_features),
            nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transition(x)

# ------------- Optimized DenseNet (ModelNew) -------------------------
class ModelNew(nn.Module):
    def __init__(self, growth_rate: int = 32, num_classes: int = 1000):
        super(ModelNew, self).__init__()

        # Initial convolution and pooling
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            FusedBatchNormReLU2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        num_features = 64
        block_layers = [6, 12, 24, 16]  # DenseNet-121 configuration

        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()

        for i, num_layers in enumerate(block_layers):
            blk = DenseBlock(num_layers, num_features, growth_rate)
            self.dense_blocks.append(blk)
            num_features += num_layers * growth_rate

            if i != len(block_layers) - 1:
                trans_out = num_features // 2
                self.transition_layers.append(TransitionLayer(num_features, trans_out))
                num_features = trans_out

        self.final_bn_relu = FusedBatchNormReLU2d(num_features)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)

        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i != len(self.dense_blocks) - 1:
                x = self.transition_layers[i](x)

        x = self.final_bn_relu(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        x = self.classifier(x)
        return x

# ----------------- Helper functions ----------------------------------
batch_size = 5
num_classes = 5
height, width = 112, 112

def get_inputs():
    return [torch.rand(batch_size, 3, height, width)]

def get_init_inputs():
    return [32, num_classes]
