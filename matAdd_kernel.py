import numpy as np
from numba import cuda, float32

# Matrix size
M, N = 512, 512

@cuda.jit
def matAdd(d_A, d_B, d_C):
    # Allocate shared memory for matrices A, B, and C
    s_A = cuda.shared.array(shape=(16, 16), dtype=float32)
    s_B = cuda.shared.array(shape=(16, 16), dtype=float32)
    s_C = cuda.shared.array(shape=(16, 16), dtype=float32)
    
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    
    # Load elements into shared memory
    row_A = by * 16 + ty
    col_A = bx * 16 + tx
    row_B = by * 16 + ty
    col_B = bx * 16 + tx
    
    s_A[ty, tx] = d_A[row_A, col_A]
    s_B[ty, tx] = d_B[row_B, col_B]
    
    cuda.syncthreads()
    
    # Perform matrix addition in shared memory
    s_C[ty, tx] = s_A[ty, tx] + s_B[ty, tx]
    
    cuda.syncthreads()
    
    # Store the result in global memory
    row_C = by * 16 + ty
    col_C = bx * 16 + tx
    d_C[row_C, col_C] = s_C[ty, tx]


if __name__ == "__main__":
    
    h_A = np.random.rand(M, N).astype(np.float32)
    h_B = np.random.rand(M, N).astype(np.float32)
    h_C = np.empty_like(h_A)
    
    # Allocate device memory
    d_A = cuda.to_device(h_A)
    d_B = cuda.to_device(h_B)
    d_C = cuda.device_array_like(h_C)
    
    # Define block and grid dimensions
    block_dim = (16, 16)
    grid_dim = ((N + block_dim[1] - 1) // block_dim[1], (M + block_dim[0] - 1) // block_dim[0])
    
    # Launch kernel
    matAdd[grid_dim, block_dim](d_A, d_B, d_C)
    
    # Copy result to host
    d_C.copy_to_host(h_C)
    
    # Display results
    print("Matrix A:")
    print(h_A)
    print("----------")
    print("Matrix B:")
    print(h_B)
    print("----------")
    print("Matrix C (Result of Addition):")
    print(h_C)
