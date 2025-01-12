#include <iostream>
#include <malloc.h>
#include <chrono>

#define BLOCK_SIZE 	256	

float sumCPU(float* vec) {
    float sum = 0;

    auto start = std::chrono::system_clock::now();
    for (int i = 0; i < N; ++i){
        sum += vec[i];
    }
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;

    std::cout << "CPU: Summ = " << sum << ", Time = " << elapsed_seconds.count() << " sec\n";
    return sum;
}

__global__ void reduce(float* inData, float* outData, int n) {
    extern __shared__ float data[];
    int i = threadIdx.x;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    data[i] = (tid < n) ? inData[tid] : 0.0f;

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (i < s) {
            data[i] += data[i + s];
        }
        __syncthreads();
    }

    if (i == 0) {
        outData[blockIdx.x] = data[0];
    }
}

float sumGPU(float* vec) {
    float *inData = nullptr, *outData = nullptr;
    float sum = 0.0f;
    int countBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaError_t cuerr = cudaMalloc(&inData, N * sizeof(float));
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "Cannot allocate device array for inData: %s\n",
            cudaGetErrorString(cuerr));
        return 1;
    }

    cuerr = cudaMalloc(&outData, countBlocks * sizeof(float));
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "Cannot allocate device array for outData: %s\n",
            cudaGetErrorString(cuerr));
        return 1;
    }

    cuerr = cudaMemcpy(inData, vec, N * sizeof(float), cudaMemcpyHostToDevice);
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "Cannot copy a from host to device: %s\n",
            cudaGetErrorString(cuerr));
        return 1;
    }   

    auto start = std::chrono::high_resolution_clock::now();
    reduce<<<countBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float) >>> (inData, outData, N);

    cuerr = cudaDeviceSynchronize();
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "Cannot synchronize CUDA kernel: %s\n",
            cudaGetErrorString(cuerr));
        return 1;
    }

    if (countBlocks > 1) {
        int newCountBlocks = (countBlocks + BLOCK_SIZE - 1) / BLOCK_SIZE;
        reduce<<<newCountBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float) >>> (outData, outData, countBlocks);
        
        cuerr = cudaDeviceSynchronize();
        if (cuerr != cudaSuccess) {
            fprintf(stderr, "Cannot synchronize CUDA kernel: %s\n",
                cudaGetErrorString(cuerr));
            return 1;
        }

        countBlocks = newCountBlocks;
    }

    cuerr = cudaMemcpy(&sum, outData, sizeof(float), cudaMemcpyDeviceToHost);
    if (cuerr != cudaSuccess) {
        fprintf(stderr, "Cannot copy b from device to host: %s\n",
            cudaGetErrorString(cuerr));
        return 1;
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "GPU: Summ = " << sum << ", Time = " << elapsed_seconds.count() << " sec\n";

    cudaFree(inData);
    cudaFree(outData);

    return sum;
}

int main() {
    float* a = (float*)calloc(N, sizeof(float));

    for (int i = 0; i < N; ++i) {
        a[i] = 1;
    }

    float cpu_sum = sumCPU(a);
    float gpu_sum = sumGPU(a);

    free(a);

    return 0;
}
