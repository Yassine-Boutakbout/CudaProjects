#include"cuda_runtime.h"
#include"device_launch_parameters.h"
#include"stdio.h"

//prototypes

__global__ void helloWorld(char*);

//host function

int main(int argc, char** argv) {
	int i;
	//desired output
	char str[] = "Hello World!";
	//mangle content of output
	for (i = 0; i < 12; i++)
	{
		//allocate memory on the device
		char* d_str;
		size_t size = sizeof(str);
		cudaMalloc((void**)&d_str, size);
		//copy the string of the device
		cudaMemcpy(d_str, str, size, cudaMemcpyHostToDevice);
		//set the grid & block sizes
		dim3 gridDim(2);//gridDim nbr of blocks in a grid -here one block per word, block 1D
		dim3 blockDim(6);//blockDim nbr threads in a block - here one thread per character
		//invoke the kernel
		helloWorld<<<gridDim, blockDim >>> (d_str);//execution of the GPU fct
		//retrieve the results from the device
		cudaMemcpy(str, d_str, size, cudaMemcpyDeviceToHost);
		//free up the allocated memory on the device
		cudaFree(d_str);
		//result
		printf("%s \n", str);
		return 0;
	}
}

//device kernel

__global__ void helloWorld(char* str) {

	//determine where in the thread grid wz are
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//unmangle output
	str[idx] += idx;
}