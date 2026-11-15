#include <iostream>
#include <math.h>
#include <chrono>

// __global__ tells the compiler this function runs on the GPU and is called by the CPU
__global__ void add(int n, float *x, float *y) {
    // Calculate which thread this is
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = x[i] + y[i];
    }
}

int main(void) {
    int N = 1<<20; // 1 Million elements (1,048,576)
    float *x, *y;

    // Allocate Unified Memory (accessible from CPU or GPU)
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    // Initialize arrays on the CPU
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    std::cout << "Data initialized. Launching GPU Kernel..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    // Launch the kernel on the GPU with 4096 blocks of 256 threads each
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    add<<<numBlocks, blockSize>>>(N, x, y);

    // Wait for the GPU to finish before the CPU continues
    cudaDeviceSynchronize();

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    // Verify the math worked (1.0 + 2.0 = 3.0)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i]-3.0f));

    std::cout << "Max error: " << maxError << std::endl;
    std::cout << "GPU Execution time: " << duration.count() << " microseconds" << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);
    
    return 0;
}
