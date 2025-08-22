import torch
   import torch.nn as nn
   from torch.utils.cpp_extension import load_inline

   source = """
   #include <torch/extension.h>
   #include <cuda_runtime.h>
   #include <math.h>

   #define N 512

   __global__ void upper_triangle_mult_kernel(const float* A, const float* B, float* C) {
       int l = blockIdx.x * blockDim.x + threadIdx.x;
       if (l >= N * (N + 1) / 2) return;

       double discriminant = (static_cast<double>(2 * N + 1) * (2 * N + 1) - 8 * l);
       double sqrt_d = sqrt(discriminant);
       double numerator = static_cast<double>(2 * N + 1) - sqrt_d;
       int i = static_cast<int>((numerator / 2.0 + 0.5));

       // Adjust i if needed
       int T_i = i * N - (i * (i - 1)) / 2;
       if (T_i > l) {
           i -= 1;
       }

       T_i = i * N - (i * (i - 1)) / 2;
       int remaining = l - T_i;
       int j = i + remaining;

       float sum = 0.0f;
       for (int k = i; k <= j; ++k) {
           sum += A[i * N + k] * B[k * N + j];
       }
       C[i * N + j] = sum;
   }

   torch::Tensor upper_triangle_mult_cuda(torch::Tensor A, torch::Tensor B) {
       const int M = N * (N + 1) / 2;
       const int block_size = 256;
       int num_blocks = (M + block_size - 1) / block_size;

       auto C = torch::zeros_like(A);

       // Check tensors are on CUDA and contiguous
       AT_ASSERTM(A.device().is_cuda(), "Input tensors must be on CUDA");
       AT_ASSERTM(B.device().is_cuda(), "Input tensors must be on CUDA");
       A = A.contiguous();
       B = B.contiguous();

       upper_triangle_mult_kernel<<<num_blocks, block_size>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>());

       // Synchronize to ensure completion
       cudaDeviceSynchronize();
       return C;
   }
   """

   cpp_src = "torch::Tensor upper_triangle_mult_cuda(torch::Tensor A, torch::Tensor B);"

   # Compile the CUDA code
   upper_triangle_mult = load_inline(
       name="upper_triangle_mult",
       cpp_sources=cpp_src,
       cuda_sources=source,
       functions=["upper_triangle_mult_cuda"],
       verbose=True
   )

   class ModelNew(nn.Module):
       def __init__(self):
           super().__init__()
           self.op = upper_triangle_mult

       def forward(self, A, B):
           return self.op.upper_triangle_mult_cuda(A, B)
