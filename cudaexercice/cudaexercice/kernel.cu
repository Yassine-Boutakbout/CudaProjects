#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include"stdio.h"

//prototypes

__global__ void mycourse(char*);

//host function
int main(int argc, char** argv) {
	//desired output
	char str[] = "my cuda course!";
	//allocate memory on the device
	char* d_str;
	size_t size = sizeof(str);
	cudaMalloc((void**)&d_str, size);

	//copy the string of the device
	cudaMemcpy(d_str, str, size, cudaMemcpyHostToDevice);

	//set the grid & block sizes
	dim3 gridDim(3);//gridDim nbr of blocks in a grid -here one block per word, block 1D
	dim3 blockDim(6);//blockDim nbr threads in a block - here one thread per character


	cudaSetDevice(0);
	//invoke the kernel
	mycourse <<<gridDim, blockDim >>> (d_str);//execution of the GPU fct

	cudaDeviceSynchronize();
	//free up the allocated memory on the device
	cudaFree(d_str);
	cudaDeviceReset();
	return 0;
}

//device kernel
__global__ void mycourse(char* str) {

	//determine where in the thread grid we are  0 1 2*5 + 0 1 2 3 4
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//print content of string
	printf("%c", str[idx]);
}