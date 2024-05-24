#include <stdio.h>
#include <cuda_runtime.h>
#include "../include/MatrixMul.cuh"

__global__
void MatrixMul(int heightA, int widthA, int widthB, float *matrixA, float *matrixB, float *matrixResult) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < heightA && col < widthB) {
        float value = 0.0;

        for (int k = 0; k < widthA; ++k) {
            value += matrixA[row * widthA + k] * matrixB[k * widthB + col];
        }

        matrixResult[row * widthB + col] = value;
    }
}

int main() {
    int heightA = 2, widthA = 3, widthB = 2;

    // Host matrices
    float h_matrixA[] = {1, 2, 3, 4, 5, 6};
    float h_matrixB[] = {7, 8, 9, 10, 11, 12};
    float h_matrixResult[4] = {0};

    float *d_matrixA, *d_matrixB, *d_matrixResult;

    cudaMalloc(&d_matrixA, heightA * widthA * sizeof(float));
    cudaMalloc(&d_matrixB, widthA * widthB * sizeof(float));
    cudaMalloc(&d_matrixResult, heightA * widthB * sizeof(float));

    cudaMemcpy(d_matrixA, h_matrixA, heightA * widthA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrixB, h_matrixB, widthA * widthB * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((widthB + blockDim.x - 1) / blockDim.x, (heightA + blockDim.y - 1) / blockDim.y);

    MatrixMul<<<gridDim, blockDim>>>(heightA, widthA, widthB, d_matrixA, d_matrixB, d_matrixResult);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();

    cudaMemcpy(h_matrixResult, d_matrixResult, heightA * widthB * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < heightA; ++i) {
        for (int j = 0; j < widthB; ++j) {
            printf("%f ", h_matrixResult[i * widthB + j]);
        }
        printf("\n");
    }

    cudaFree(d_matrixA);
    cudaFree(d_matrixB);
    cudaFree(d_matrixResult);

    return 0;
}
