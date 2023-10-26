import numpy as np
from numba import cuda, float32

@cuda.jit
def matMul(d_A, d_B, d_C):
    row, col = cuda.grid(2)
    
    if row < d_C.shape[0] and col < d_C.shape[1]:
        M, N = d_A.shape
        P = d_B.shape[1]
        sum = 0.0
        
        for i in range(N):
            sum += d_A[row, i] * d_B[i, col]
        
        d_C[row, col] = sum

if __name__ == "__main__":
    
    M, N, P = 3, 4, 2
    
    
    h_A = np.random.randint(1, 10, size=(M, N)).astype(np.float32)
    h_B = np.random.randint(1, 10, size=(N, P)).astype(np.float32)
    h_C = np.empty((M, P), dtype=np.float32)
    
    # Allocate device memory
    d_A = cuda.to_device(h_A)
    d_B = cuda.to_device(h_B)
    d_C = cuda.device_array_like(h_C)
    
    block_dim = (16, 16)
    grid_dim = ((P + block_dim[1] - 1) // block_dim[1], (M + block_dim[0] - 1) // block_dim[0])
    
    matMul[grid_dim, block_dim](d_A, d_B, d_C)
    
    # Copy result to host
    d_C.copy_to_host(h_C)
    
    
    print("Matrix A:")
    print(h_A)
    print("--------")
    print("Matrix B:")
    print(h_B)
    print("--------")
    print("Matrix C (Result of Multiplication):")
    print(h_C)
