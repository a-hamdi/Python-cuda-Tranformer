import numpy as np
from numba import cuda

@cuda.jit
def  batch_matrix_multiplication(d_A, d_B, d_C, batch_size, M, N, P):
    row, col, batch = cuda.grid(3)
    
    if batch < batch_size and row < M and col < P:
        sum = 0.0
        
        for i in range(N):
            sum += d_A[batch, row, i] * d_B[batch, i, col]
        
        d_C[batch, row, col] = sum

if __name__ == "__main__":
    M = 2
    N = 3
    P = 5
    batch_size = 4

    h_A = np.random.randint(1, 11, size=(batch_size, M, N)).astype(np.float32)
    h_B = np.random.randint(1, 11, size=(batch_size, N, P)).astype(np.float32)
    h_C = np.empty((batch_size, M, P), dtype=np.float32)

    # Allocate device memory
    d_A = cuda.to_device(h_A)
    d_B = cuda.to_device(h_B)
    d_C = cuda.device_array_like(h_C)

    # Define block and grid dimensions
    block_dim = (8, 8, batch_size)
    grid_dim = ((P + block_dim[0] - 1) // block_dim[0], (M + block_dim[1] - 1) // block_dim[1], batch_size)

    batch_matrix_multiplication[grid_dim, block_dim](d_A, d_B, d_C, batch_size, M, N, P)

    # Copy result to host
    d_C.copy_to_host(h_C)

    for batch in range(batch_size):
        print(f"Matrix A (Batch {batch + 1}):\n", h_A[batch])
        print(f"Matrix B (Batch {batch + 1}):\n", h_B[batch])
        print(f"Matrix C (Batch {batch + 1}):\n", h_C[batch])
        print("--------")

