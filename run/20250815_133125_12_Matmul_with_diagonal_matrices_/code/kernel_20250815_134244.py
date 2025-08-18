cuda
__global__ void diag_mult_optimized(const float* A,const float* B, float* C, int N, int M) {
    int row = blockIdx.x;
    if (row >= N) return;
    int tid = threadIdx.x;

    __shared__ float a_val;
    if (tid ==0) {
        a_val = A[row];
    }
    __syncthreads();

    // Now, each thread processes some number of columns in the row.
    // Each thread can process M / threads_per_block columns (rounded up).
    // For M elements, each thread processes M / blockDim.x elements per row.
    int start_j = tid * (M / blockDim.x); // Not exactly correct, but approximate.

    for (int j = tid * (M / blockDim.x); j < M; j += blockDim.x) { 
        // Or using a step of blockDim.x, but need to distribute M elements among threads in block.
        int idx = row * M + j;
        C[idx] = a_val * B[idx];
    }
}
