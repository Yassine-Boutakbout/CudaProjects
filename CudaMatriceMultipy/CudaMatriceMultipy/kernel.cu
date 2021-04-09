#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include"stdio.h"

__global__ void matrixMult(int* a, int* b, int* c, int width) {
	int i, sum = 0;
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int row = threadIdx.y + blockDim.y * blockIdx.y;
	if (col < width && row < width) {
		for (i = 0; i < width; i++) {
			sum += a[row * width + i] * b[i * width + col];
		}
		c[row * width + col] = sum;
	}
}

int main(int argc, char* argv[]) {
	int N = 3;
	//int a[N][N], b[N][N], c[N][N];
	int* h_a, * h_b, * h_c;
	int* dev_a, * dev_b, * dev_c;
	int i, j;
	int size = N * N * sizeof(int);
	cudaMallocHost((void**)&h_a, size);
	cudaMallocHost((void**)&h_b, size);
	cudaMallocHost((void**)&h_c, size);

	// initialize matrices a and b with appropriate values
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			h_a[i * N + j] = i + j;
			h_b[i * N + j] = i * j;

		}
	}
	printf(" ****matrice 1***** \n");
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			printf("%d\t", h_a[i * N + j]);

		}
		printf("\n");
	}
	printf(" ****matrice 2*****\n");
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			printf("%d\t", h_b[i * N + j]);
		}
		printf("\n");
	}

	cudaMalloc((void**)&dev_a, size);
	cudaMalloc((void**)&dev_b, size);
	cudaMalloc((void**)&dev_c, size);

	cudaMemcpy(dev_a, h_a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, h_b, size, cudaMemcpyHostToDevice);

	dim3 gridDim(1, 1);
	dim3 blockDim(N, N);

	matrixMult <<<gridDim, blockDim >>> (dev_a, dev_b, dev_c, N);

	cudaMemcpy(h_c, dev_c, size, cudaMemcpyDeviceToHost);
	printf("****Résultat****\n");
	for (i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			printf("%d\t", h_c[i * N + j]);
		}
		printf("\n");
	}

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	cudaFreeHost(h_a);
	cudaFreeHost(h_b);
	cudaFreeHost(h_c);

	return 0;
}