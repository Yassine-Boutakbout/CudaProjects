#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdio.h"
#include "malloc.h"

__global__ void vector(int* X, int* Y,int* Z,int sc, int N) {

	int idx= blockDim.x * blockIdx.x + threadIdx.x;

	if (idx < N)
	{
		Z[idx] = sc * X[idx] + Y[idx];
		printf("Z[%d]=%d \n", idx, Z[idx]);
	}
}

int main() {

	int N = 16, sc = 2;
	size_t size = N * sizeof(int);

	//Allocate input vectors h_X and h_Y in host memory
	int* H_X = (int*)malloc(size);
	int* H_Y = (int*)malloc(size);
	int* H_Z = (int*)malloc(size);

	// Initialize input vectors
	for (int i = 0; i < N; i++)
	{
		H_X[i] = 1;
		H_Y[i] = 2;
	}

	int *D_X,*D_Y,*D_Z;

	// Allocate vectors in device memory
	cudaMalloc(&D_X, size);
	cudaMalloc(&D_Y, size);
	cudaMalloc(&D_Z, size);


	// Copy vectors from host memory to device memory
	cudaMemcpy(D_X, H_X, size, cudaMemcpyHostToDevice);
	cudaMemcpy(D_Y, H_Y, size, cudaMemcpyHostToDevice);

	int threads = 8;
	int blocks = (N+7)/ 8;

	//set the grid & block sizes
	dim3 gridDim(blocks);//gridDim nbr of blocks in a grid
	dim3 blockDim(threads);//blockDim nbr threads in a block

	//Invoke kernel
	vector <<<gridDim, blockDim >>> (D_X, D_Y, D_Z, sc, N);

	// Copy result from device memory to host memory
	cudaMemcpy(H_Z, D_Z, size, cudaMemcpyDeviceToHost);

	// Free device memory
	cudaFree(D_X);
	cudaFree(D_Y);
	cudaFree(D_Z);

	// Free host memory
	free(H_X);
	free(H_Y);
	free(H_Z);
}