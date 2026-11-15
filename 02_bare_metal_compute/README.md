# 02: Bare-metal compute

CUDA written directly, with no framework between the code and the GPU. So far: one kernel, and a measurement of why its first timing was misleading.

## vector_add.cu

Adds two arrays of 1,048,576 floats on the GPU. One thread handles one element (`i = blockIdx.x * blockDim.x + threadIdx.x`), launched as 4096 blocks of 256 threads. The arrays come from `cudaMallocManaged`, so the same pointers work on the CPU and the GPU and there are no explicit copies. The CPU then checks every element against 1 + 2 = 3.

It prints a GPU execution time of about 1.5 ms. That number is not the kernel.

## vector_add_timing.cu

Runs the same kernel three ways on the same arrays.

| | What is timed | Median of 15 runs |
|---|---|---|
| A | launch + sync, arrays last written by the CPU | 1528 µs |
| B | launch + sync, arrays prefetched to the GPU first | 13 µs |
| C | the kernel alone (CUDA events), arrays already on the GPU | 10 µs |

With cold data the GPU faults the pages across from CPU memory while the kernel runs, and that migration is roughly 99% of what `vector_add.cu` prints. The kernel itself takes about 10 µs. Timing a managed-memory kernel without prefetching measures the migration, not the kernel.

## Build and run

Needs an NVIDIA GPU and the CUDA toolkit.

```bash
nvcc -O2 -o vector_add vector_add.cu && ./vector_add
nvcc -O2 -o vector_add_timing vector_add_timing.cu && ./vector_add_timing
```

Measured on an RTX 4070 Laptop GPU (8 GB), driver 580.178.04, nvcc 11.5. Both programs report a max error of 0.

## Not done yet

- No run on an edge board such as a Jetson.
- No comparison against the same operation in PyTorch or NumPy.
- One kernel. The aim is GPU code for high-frequency control loops, and vector add is the warm-up.
