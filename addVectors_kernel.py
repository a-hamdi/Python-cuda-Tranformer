import numpy as np
from numba import cuda, float32

@cuda.jit
def addVectors(d_a, d_b, d_c):
    col = cuda.grid(1)
    
    if col < d_c.shape[0]:
        d_c[col] = d_a[col] + d_b[col]

if __name__ == "__main__":
    N = 10
    
    h_a = np.arange(N, dtype=np.float32) - 3
    h_b = np.arange(N, dtype=np.float32)
    h_c = np.empty(N, dtype=np.float32)
    
    # Allocate device memory
    d_a = cuda.to_device(h_a)
    d_b = cuda.to_device(h_b)
    d_c = cuda.device_array_like(h_c)

    block_dim = 256
    grid_dim = (N + block_dim - 1) // block_dim

    addVectors[grid_dim, block_dim](d_a, d_b, d_c)
    
    # Copy result to host
    d_c.copy_to_host(h_c)

    for i in range(10):
        print(f'a: {h_a[i]} b: {h_b[i]} c: {h_c[i]}')
    


