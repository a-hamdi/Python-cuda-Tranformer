import numpy as np
from numba import cuda, float32

@cuda.jit
def matScale(d_A, d_B, scale):
    row, col = cuda.grid(2)
    
    if row < d_B.shape[0] and col < d_B.shape[1]:
        d_B[row, col] = d_A[row, col] / scale

if __name__ == "__main__":
    
    M, N = 4, 4
    
    
    scale = 2.5
    
    
    h_A = np.random.randint(1, 10, size=(M, N)).astype(np.float32)
    h_B = np.empty_like(h_A)
    
    # Allocate device memory
    d_A = cuda.to_device(h_A)
    d_B = cuda.to_device(h_B)
    
    # Define block and grid dimensions
    block_dim = (16, 16)
    grid_dim = ((N + block_dim[1] - 1) // block_dim[1], (M + block_dim[0] - 1) // block_dim[0])
    
    matScale[grid_dim, block_dim](d_A, d_B, scale)
    
    # Copy result to host
    d_B.copy_to_host(h_B)
    
    # Display matrices and result
    print("Matrix A:")
    print(h_A)
    print("--------")
    print("Matrix B (Result of Scaling):")
    print(h_B)
