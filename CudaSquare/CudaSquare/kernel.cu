#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdio.h"
#include "malloc.h"


__global__ void kernel(int* A, int N) {

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < N)
		A[idx] = A[idx] * A[idx];
}

int main() {

	int N = 3;
	size_t size = N * sizeof(int);

	//allocation input vector
	int* H_A = (int*)malloc(N * sizeof(int));


	// Initialize input vectors
	for (int i = 0; i < N; i++)
	{
		H_A[i] = i;
	}

	int* D_A;

	// Allocate vectors in device memory
	cudaMalloc(&D_A, size);

	// Copy vector from host memory to device memory
	cudaMemcpy(D_A, H_A, size, cudaMemcpyHostToDevice);

	int threads = 8;
	int blocks = (N + 7) / 8;

	//set the grid & block sizes
	dim3 gridDim(blocks);//gridDim nbr of blocks in a grid
	dim3 blockDim(threads);//blockDim nbr threads in a block

	//Invoke kernel
	kernel << <gridDim, blockDim >> > (D_A,N);

	// Copy result from device memory to host memory
	cudaMemcpy(H_A, D_A, size, cudaMemcpyDeviceToHost);

	for (int j = 0; j < N; j++)
	{
		printf("A[%d]=%d \n", j, H_A[j]);
	}

	cudaFree(D_A);
	free(H_A);
}