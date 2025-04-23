#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1024 // Tamaño del vector

__global__ void gradientCalculation(float *x, float *grad) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        grad[tid] = 2 * x[tid]; // Cálculo del gradiente de f(x) = x^2
    }
}

int main() {
    float *x, *grad;
    float *d_x, *d_grad;
    int size = N * sizeof(float);

    // Alojar memoria en el host
    x = (float*)malloc(size);
    grad = (float*)malloc(size);

    // Inicializar valores de x
    for (int i = 0; i < N; ++i) {
        x[i] = i;
    }

    // Inicializar valores de x
    for (int i = 0; i < N; ++i) {
        printf("%f\n",x[i]);
    }

    // Alojar memoria en el dispositivo (GPU)
    cudaMalloc(&d_x, size);
    cudaMalloc(&d_grad, size);

    // Copiar datos desde el host al dispositivo
    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);

    // Definir el número de hilos por bloque y bloques por grid
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Llamar al kernel para calcular el gradiente
    gradientCalculation<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_grad);

    // Comprobar errores de CUDA
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "Error en el kernel: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }


    // Copiar el resultado desde el dispositivo al host
    cudaMemcpy(grad, d_grad, size, cudaMemcpyDeviceToHost);

 // Comprobar errores de CUDA
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "Error en la copia de memoria: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Imprimir el gradiente calculado
    printf("Gradiente de f(x) = x^2:\n");
    for (int i = 0; i < N; ++i) {
        printf("grad[%d] = %f\n", i, grad[i]);
    }

    // Liberar memoria en el dispositivo
    cudaFree(d_x);
    cudaFree(d_grad);

    // Liberar memoria en el host
    free(x);
    free(grad);

    return 0;
}
