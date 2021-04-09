#include"cuda_runtime.h"
#include"device_launch_parameters.h"
#include"stdio.h"
#include <malloc.h>
#define N (1024*1024)
#define M (5000)

__global__ void cudakernel(float* buf)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	buf[i] = 1.0f * i / N;
	for (int j = 0; j < M; j++) { buf[i] = buf[i] * buf[i] - 0.25f; }
}

int main()
{
	float *data;
	float *d_data;
	data = (float*)malloc(N*sizeof(float));
	cudaMalloc(&d_data, N * sizeof(float));
	cudaMemcpy(d_data, data, N * sizeof(float), cudaMemcpyHostToDevice);
	cudakernel<<< N/256, 256 >>>(d_data);
	cudaMemcpy(data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_data);

	int sel;
	sel = 1000;
	printf("data[%d]=%f \n", sel, data[sel]);
}