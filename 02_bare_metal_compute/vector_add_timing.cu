// Why does vector_add.cu report about 1.6 ms for a kernel that should take microseconds?
// This program times the same add kernel three ways on the same 1M-element arrays.
//
//   A  cold:        the CPU wrote the arrays last, so the GPU has to pull the pages over
//                   during the kernel (unified memory migrates them on demand).
//                   Wall clock around launch + sync, like vector_add.cu.
//   B  prefetched:  the same wall-clock measurement after cudaMemPrefetchAsync has moved
//                   the arrays to the GPU up front.
//   C  kernel only: CUDA events around the launch, arrays already resident on the GPU.
//
// Build: nvcc -O2 -o vector_add_timing vector_add_timing.cu
#include <iostream>
#include <math.h>
#include <chrono>

__global__ void add(int n, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = x[i] + y[i];
    }
}

static void fill(int n, float *x, float *y) {
    for (int i = 0; i < n; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }
}

// Wall-clock time of one launch plus the wait for it to finish, in microseconds.
static long long launch_wall_us(int n, float *x, float *y, int blocks, int threads) {
    auto start = std::chrono::high_resolution_clock::now();
    add<<<blocks, threads>>>(n, x, y);
    cudaDeviceSynchronize();
    auto stop = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
}

int main(void) {
    const int N = 1 << 20;
    const size_t bytes = N * sizeof(float);
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    const int device = 0;

    float *x, *y;
    cudaMallocManaged(&x, bytes);
    cudaMallocManaged(&y, bytes);

    // Pay the one-time launch costs (module load, context setup) up front.
    // n = 0, so this launch touches no memory and migrates nothing.
    add<<<1, 1>>>(0, x, y);
    cudaDeviceSynchronize();

    // A: cold. The CPU wrote the arrays, so the GPU faults the pages in during the kernel.
    fill(N, x, y);
    long long cold_us = launch_wall_us(N, x, y, blocks, threads);

    // B: prefetched. Rewrite on the CPU, then move everything to the GPU before timing.
    fill(N, x, y);
    cudaMemPrefetchAsync(x, bytes, device);
    cudaMemPrefetchAsync(y, bytes, device);
    cudaDeviceSynchronize();
    long long prefetched_us = launch_wall_us(N, x, y, blocks, threads);

    // C: kernel only. The arrays are already on the GPU from B, so time just the kernel.
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    add<<<blocks, threads>>>(N, x, y);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float kernel_ms = 0.0f;
    cudaEventElapsedTime(&kernel_ms, t0, t1);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // B left y = 1 + 2 = 3 and C added x = 1 once more, so every y[i] should be 4.
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i] - 4.0f));

    std::cout << "Max error: " << maxError << std::endl;
    std::cout << "A cold (launch + sync):       " << cold_us << " us" << std::endl;
    std::cout << "B prefetched (launch + sync): " << prefetched_us << " us" << std::endl;
    std::cout << "C kernel only (CUDA events):  " << (long long)(kernel_ms * 1000.0f) << " us" << std::endl;

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaFree(x);
    cudaFree(y);
    return 0;
}
