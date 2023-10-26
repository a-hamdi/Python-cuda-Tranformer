# Overview

Welcome to this repository showcasing a suite of CUDA-accelerated mathematical operations implemented entirely in Python. These programs harness the immense computational power of NVIDIA GPUs to perform matrix and vector operations efficiently and swiftly. 
The kernels I implemented are the building blocks for the transformer model and I believe it's safe to say This is a Python-cuda Tranformer.

# Technologies and Frameworks

- CUDA
- Python
- Nvidia CUDA kernel
- Parallel reduction
- Parallel processing
- GPU programming
- Matrix and vector operations
- Dot product calculation
- Relu
- Softmax function


# Installation

This guide will walk you through the steps required to install and run the project.

## Prerequisites

**NVIDIA GPU**: A compatible NVIDIA GPU is required as CUDA relies on GPU processing power. The specific GPU requirements might vary based on the complexity of the operations and the size of the data you're dealing with.

**CUDA Toolkit**: You need to have the CUDA Toolkit installed on your system. The toolkit includes libraries and tools for CUDA development. Make sure the version you install is compatible with your GPU and your Python environment.

**Python Environment**: An appropriate Python environment is essential. You might be using Anaconda, virtual environments, or a specific Python version. Ensure that the required packages (like Numba and NumPy) are installed.

**Numba and NumPy**: These are Python libraries crucial for CUDA programming. Numba provides a Just-In-Time compiler for Python functions, which is essential for CUDA programming. NumPy, on the other hand, is fundamental for numerical operations in Python.

## How to Use:

. **Clone the repository**

   Clone the repository using the following command:

   ```
   git clone https://github.com/username/repository.git
   ```

6. **Run the CUDA Kernels**

   you can run them using the following command:

   ```
   ./matMuldot_product_kernel
   ```





# Example



## dot_product_kernel.py

The `dot_product_kernel.py` script is used for applying a dot product.

```py
# Initialize vectors A and B
# specify the size of the vectors
A = np.arange(size, dtype=np.float32)
B = np.arange(size, dtype=np.float32)

dotProd(A,B,result,size)

```



If you encounter any issues or have concerns, please don't hesitate to reach out to me directly. Your feedback is valuable!