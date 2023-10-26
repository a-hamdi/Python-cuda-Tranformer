import numpy as np
from numba import cuda, float32

@cuda.jit
def relu(d_a, d_b, alpha):
    col = cuda.grid(1)
    
    if col < d_b.shape[0]:
        d_b[col] = max(alpha * d_a[col], d_a[col])


if __name__ == "__main__":
    # Vector size
    N = 8
    
    # Leaky ReLU parameter
    alpha = 0.01
    
    
    h_a = np.random.randint(-5, 8, size=N).astype(np.float32)
    h_b = np.empty_like(h_a)
    
    # Allocate device memory
    d_a = cuda.to_device(h_a)
    d_b = cuda.to_device(h_b)
  
    block_dim = 256
    grid_dim = (N + block_dim - 1) // block_dim
    
    relu[grid_dim, block_dim](d_a, d_b, alpha)
    
    # Copy result to host
    d_b.copy_to_host(h_b)
    
    print("Vector A:")
    print(h_a)
    print("\nLeaky ReLU:")
    print(h_b)
