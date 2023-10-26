import numpy as np
from numba import cuda

@cuda.jit
def transpose_kernel(d_A, d_T, M, N):
    row, col = cuda.grid(2)

    if row < M and col < N:
        d_T[col, row] = d_A[row, col]

if __name__=="__main__":

  M = 4
  N = 4

  # Initialize host data
  h_A = np.random.randint(1, 11, size=(M, N)).astype(np.float32)
  h_T = np.empty((N, M), dtype=np.float32)

  # Allocate device memory
  d_A = cuda.to_device(h_A)
  d_T = cuda.to_device(h_T)

  block_dim = (16, 16)
  grid_dim = ((N + block_dim[0] - 1) // block_dim[0], (M + block_dim[1] - 1) // block_dim[1])

  transpose_kernel[grid_dim, block_dim](d_A, d_T, M, N)

  # Copy result back to host
  d_T.copy_to_host(h_T)

  print("Matrix A:")
  print(h_A)
  print("Transpose:")
  print(h_T)
