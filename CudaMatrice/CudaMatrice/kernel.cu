#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdio.h"
#include "malloc.h"


const int N = 2;
const int blocksize = 1;
const int MAX = 100;

__host__ void add_matrix_cpu(float* a, float* b, float* c, int N) {
	int i, j;
	printf("Host \n");
	for (i = 0; i < N; i++)
	{
		printf("\n");
		for (j = 0; j < N; j++)
		{
			c[i * N + j] = a[i * N + j] + b[i * N + j];
			printf("C[%d]=%f \t",i*N+1,c[i * N + 1]);
		}
	}
}

__global__ void add_matrix(float* a, float* b, float* c, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int index = i + j * N;
	if (i < N && j < N)
		c[index] = a[index] + b[index];
}

int main() {
	int k;
	float* a = new float[N * N];
	float* b = new float[N * N];
	float* c = new float[N * N];


	for (int i = 0; i < N * N; ++i) {
		a[i] = 1.0f; b[i] = 3.5f;
	}
	float* ad, * bd, * cd;
	const int size = N * N * sizeof(float);
	cudaMalloc((void**)&ad, size);
	cudaMalloc((void**)&bd, size);
	cudaMalloc((void**)&cd, size);
	cudaMemcpy(ad, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(bd, b, size, cudaMemcpyHostToDevice);
	dim3 dimBlock(blocksize, blocksize);
	dim3 dimGrid(N / dimBlock.x, N / dimBlock.y);

	add_matrix <<<dimGrid, dimBlock >>> (ad, bd, cd, N);
	cudaMemcpy(c, cd, size, cudaMemcpyDeviceToHost);
	cudaFree(ad);
	cudaFree(bd);
	cudaFree(cd);
	printf("Device \n");
	for (int m = 0; m < 4; m++)
	{
		if (m % 2 == 0)
		{
			printf("\n");
		}
		printf("c[%d]=%f \t",m,c[m]);
	}
	

	for (k = 1; k <= MAX; k++)
		add_matrix_cpu(a, b, c, N);
	delete[] a;
	delete[] b;
	delete[] c;
}