import numpy as np
from numba import cuda, float32
import math

@cuda.jit(device=True)
def block_reduce(a):
    # Perform parallel reduction within a block
    shared_val = cuda.shared.array(32, dtype=float32)
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    bdim = cuda.blockDim.x

    shared_val[tid] = a
    cuda.syncthreads()

    # Perform reduction in shared memory
    for s in range(bdim // 2):
        if tid < bdim // 2:
            shared_val[tid] += shared_val[tid + s + 1]
        cuda.syncthreads()

    return shared_val[0]

@cuda.jit
def mean_var_norm(d_a, d_mean, d_var, d_norm, N):
    col = cuda.threadIdx.x
    stride = cuda.blockDim.x

    # Compute mean using parallel reduction(PR)
    mean_val = 0.0
    for i in range(col, N, stride):
        mean_val += d_a[i]
    
    mean_val = cuda.atomic.add(d_mean, 0, mean_val)  
    cuda.syncthreads()

    # Compute variance using PR
    var_val = 0.0
    for i in range(col, N, stride):
        var_val += (d_a[i] - d_mean[0]) ** 2
    
    var_val = cuda.atomic.add(d_var, 0, var_val)  
    cuda.syncthreads()

    # Synchronize to make sure mean and variance are calculated
    cuda.syncthreads()

    # Compute normalized values
    for i in range(col, N, stride):
        d_norm[i] = (d_a[i] - d_mean[0]) / (math.sqrt(d_var[0] / N + 1e-8))



if __name__ == "__main__":
    N = 5
    
    # Allocate host memory
    h_a = np.random.randint(1, 11, N).astype(np.float32)
    h_mean = np.zeros(1, dtype=np.float32)
    h_var = np.zeros(1, dtype=np.float32)
    h_norm = np.zeros(N, dtype=np.float32)
    
    # Allocate device memory
    d_a = cuda.to_device(h_a)
    d_mean = cuda.device_array(1, dtype=np.float32)
    d_var = cuda.device_array(1, dtype=np.float32)
    d_norm = cuda.device_array(N, dtype=np.float32)
    
    # Launch kernel
    block_size = 32 
    grid_size = (N + block_size - 1) // block_size
    mean_var_norm[grid_size, block_size](d_a, d_mean, d_var, d_norm, N)
    
    # Copy results to host
    d_mean.copy_to_host(h_mean)
    d_var.copy_to_host(h_var)
    d_norm.copy_to_host(h_norm)
    
    
    print("A:", h_a)
    print("Mean:", h_mean[0])
    print("Variance:", h_var[0])
    print("Normalized Values:", h_norm)
