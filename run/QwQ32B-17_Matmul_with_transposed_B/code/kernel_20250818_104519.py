import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TX 16
#define TY 16
#define TW 16

__global__ void matmul_kernel(float* C, const float* A, const float* B, int M, int N, int K) {
    __shared__ float ssa[TX][TW];
    __shared__ float ssb[TW][TY];

    int block_row = blockIdx.x * TX;
    int block_col = blockIdx.y * TY;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float sum = 0.0f;

    for (int it = 0; it < (K + TW - 1) / TW; ++it) {
        int k_start = it * TW;

        // Load ssa
        if (block_row + tx < M && k_start + ty < K) {
            ssa[tx][ty] = A[(block_row + tx) * K + (k_start + ty)];
        } else {
            ssa[tx][ty] = 0.0f;
        }

        // Load ssb
        int row_B = block_col + ty;
        int col_B = k_start + tx;
        if (row_B < N && col_B < K) {
            ssb[tx][ty] = B[row_B * K + col_B];
        } else {
            ssb[tx][ty] = 0.0f;
        }

        // Synchronize
        __syncthreads();

        // Compute partial sum
        for (int k = 0; k < TW; ++k) {
            sum += ssa[tx][k] * ssb[k][ty];
        }

        // Synchronize again
        __syncthreads();
    }

    // Write output
    int row = block_row + tx;
    int col = block_col + ty;
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // Ensure A and B are on the same device
    auto device = A.device();
    assert(B.device() == device);
    
    // Get dimensions
    int M = A.size(0);
    int K_A = A.size(1);
    int N_B = B.size(0);
    int K_B = B.size(1);

    assert(K_A == K_B);

    // Create output tensor
    auto options = A.options();
    auto C = torch::empty({M, N_B}, options);

    // Launch kernel
    dim3 threads(TX, TY);
    dim3 blocks(
        (M + TX - 1) / TX,
        (N_B + TY - 1) / TY
    );

    matmul_kernel<<<blocks, threads>>>(
        C.data_ptr<float>(),
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        M, N_B, K_B // since K_A == K_B
    );

    return C;
}
"""

cpp_src = "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"

matmul_cuda = load_inline(
    name="matmul_cuda",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["matmul_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.matmul_cuda = matmul_cuda

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_cuda(A, B).cuda()  # Ensure it's on CUDA?

Wait wait, need to make sure the inputs are on the GPU. The original code's get_inputs() may create tensors on CPU, so need to ensure they are moved to CUDA before calling the kernel.

Wait, in the provided user code, the original 'get_inputs' has:

def get_inputs():
    A = torch.rand(M, K)
    B = torch.rand(N, K)
    return [A, B]

These tensors are CPU tensors by default. The user’s original code’s forward may use .cuda() somewhere, but in the task instructions says to generate the new ModelNew such that it replaces all PyTorch operators, but the rest must be unchanged.

So, in the ModelNew's forward, the inputs are expected to be tensors (presumably on the correct device). Since the kernel is written for CUDA, the tensors must be on the GPU.

Therefore, in the model's forward, it should ensure the inputs are on the device before calling the CUDA kernel.

Wait the code for the kernel's host function (matmul_cuda) also requires the tensors to be on CUDA. Thus, the user must ensure that the inputs are on the correct device. The ModelNew’s forward code can call self.matmul_cuda(A, B) but A and B must be on CUDA. So the original user code may have the model’s tensors created on the GPU.

Alternatively, the code can move them to the device inside the forward. But the problem says to leave non-replaced parts unchanged. The original forward takes tensors as inputs, so the new forward must accept the same (maybe on CPU, but in that case it won't work).

Hmm. The problem says to replace the operators inside the class Model with CUDA kernels. The user’s original code’s forward is:

return torch.matmul(A, B.T)

Thus, the new forward must replace that with self.matmul_cuda(A, B), and the kernel must compute the same as torch.matmul(A, B.T). 

The Kernel function's matmul_cuda takes B as per the problem input (NxK), so yes.

Therefore, assuming that in the user's code, the inputs are already on the GPU (the get_inputs may have .cuda()), the ModelNew's forward is simply:

def forward(self, A, B):
    return self.matmul_cuda(A, B)

But in the Python example given in the problem, the original model's forward's inputs are tensors (presumably on CPU?), but since the kernel requires CUDA tensors, perhaps the user must first move them to CUDA. 

But the problem instructions allow us to leave other parts unchanged. The user’s get_inputs may return CPU tensors, so the kernel won't work. Therefore, the 'ModelNew' must make sure tensors are on the correct device, but since the task says to leave non-replaced parts unchanged, perhaps not.

Wait the problem says:

"Leave any non-replaced parts of the model unchanged."

The original Model has a forward that uses torch.matmul(A, B.T), which is replaced with the CUDA kernel. The rest (like __init__, input/outputs) must be kept the same, so the forward's A and B should be the same tensors (so their device is same?).

The original user code does not have any .cuda() in their code, except maybe in get_inputs. Looking at the example given in the problem's user's code:

The original user's 'Model' 's get_inputs is:

def get_inputs():
    A = torch.rand(M, K)
    B = torch.rand(N, K)
    return [A, B]

These are CPU tensors. So when the user uses them in the model's forward, they would have to be on the device. But the problem's example shows that in the optimised code that uses a CUDA kernel, the ModelNew's forward must have the kernels work on CUDA tensors, so the user's code will have to move the tensors to CUDA before passing.

As per the problem's few shot example, the optimised code's forward has the tensors passed directly, with .cuda() in the wrapper's input (probably not).

But for the ModelNew, since the kernel is a CUDA kernel, the inputs A and B passed to it must be CUDA tensors. Thus, in the forward, the code is correct as long the inputs are on CUDA.

Thus, assuming that the code is run with tensors already on CUDA (e.g., in the test code), the forward code would work.

Putting all together:

The final code would have:

class ModelNew(nn.Module):

    def __init__(self):
        super().__init__()
        self.matmul_cuda = matmul_cuda  # the loaded function

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_cuda(A, B)

But in the kernel function's host code, we are already assuming the inputs are on CUDA because the kernel expects CUDA pointers.

Thus, the code should work.

Now, potential issues:

- Are the shared arrays ssa and ssb correctly declared? Yes, with proper #defines.

- Are the block and grid dimensions correctly calculated? Yes.

- Are the loops over K tiles correct? The total K chunks is ceil(K / TW).

- The dimensions of the output tensor in the device function are correct? Yes, {M, N_B}.

- The input A and B are on the same device? The assert checks that.

- The kernel's launching.

Yes.

Thus, the code should be correct.
</think>
