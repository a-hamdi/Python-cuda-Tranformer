import numpy as np
from numba import cuda, float32

@cuda.jit
def softmax(d_in, d_out):
    col = cuda.grid(1)
    
    if col < d_out.shape[0]:
        N = d_out.shape[0]
        
        local_exp = math.exp(d_in[col])
        
        # Allocate shared memory for parallel reduction
        shared_exp = cuda.shared.array(256, dtype=float32)
        
        # Store exp(x) in shared memory
        shared_exp[cuda.threadIdx.x] = local_exp
        
        # Synchronize!
        cuda.syncthreads()
        
        # Parallel reduction to compute sum(exp(x))
        for stride in range(cuda.blockDim.x // 2, 0, -1):
            if cuda.threadIdx.x < stride:
                shared_exp[cuda.threadIdx.x] += shared_exp[cuda.threadIdx.x + stride]
            
            # Synchronize threads after each step of reduction
            cuda.syncthreads()
        
        # Calculate softmax for the current element
        d_out[col] = local_exp / shared_exp[0]

# Main function
if __name__ == "__main__":
    # Vector size
    N = 8
    
    # Generate random input vector
    h_in = np.random.randint(1, 8, size=N).astype(np.float32)
    h_out = np.empty_like(h_in)
    
    # Allocate device memory
    d_in = cuda.to_device(h_in)
    d_out = cuda.to_device(h_out)

    block_dim = 256
    grid_dim = (N + block_dim - 1) // block_dim

    softmax[grid_dim, block_dim](d_in, d_out)
    
    # Copy result to host
    d_out.copy_to_host(h_out)

    print("Softmax input:")
    print(h_in)
    print("Softmax output:")
    print(h_out)
