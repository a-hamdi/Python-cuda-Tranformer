import numpy as np
from numba import cuda, float32

@cuda.jit
def dotProd(d_a, d_b, d_dotprod, N):
    sdata = cuda.shared.array(32, dtype=float32)  # Shared memory for parallel reduction
    
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x
    
    start = tx
    stride = bw
    
    dot_product = 0.0
    
    # Compute element-wise products and perform parallel reduction in shared memory
    while start < N:
        dot_product += d_a[start] * d_b[start]
        start += stride
    
    sdata[tx] = dot_product
    
    # Synchronize !!
    cuda.syncthreads()
    
    # parallel reduction in shared memory
    i = cuda.blockDim.x // 2
    while i != 0:
        if tx < i:
            sdata[tx] += sdata[tx + i]
        cuda.syncthreads()
        i //= 2
    
    # Store the final result in global memory
    if tx == 0:
        d_dotprod[bx] = sdata[0]


if __name__ == "__main__":
    N = 5
    
    # Allocate host memory
    h_a = np.random.randint(1, 11, N).astype(np.float32)
    h_b = np.random.randint(1, 11, N).astype(np.float32)
    h_dotprod = np.zeros(1, dtype=np.float32)
    
    # Allocate device memory
    d_a = cuda.to_device(h_a)
    d_b = cuda.to_device(h_b)
    d_dotprod = cuda.device_array(1, dtype=np.float32)
    
    # Launch kernel
    block_size = 32  
    grid_size = (N + block_size - 1) // block_size
    dotProd[grid_size, block_size](d_a, d_b, d_dotprod, N)
    
    # Copy result to host
    d_dotprod.copy_to_host(h_dotprod)
    
    # Print result
    print("A:", h_a)
    print("B:", h_b)
    print("Dot Product:", h_dotprod[0])
