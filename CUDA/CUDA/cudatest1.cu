#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdio.h"

__global__ void test() {
	//Empty Kernel
	printf("hello from CUDA");
}

int main() {
	test << <1, 1 >> > ();
	return 0;
}