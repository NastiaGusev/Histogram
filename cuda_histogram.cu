#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "myProto.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

__global__ void addCalculateKernel(const int *arr, int* globalHist, unsigned int size)
{
	int tid = threadIdx.x;
	int blockId = blockIdx.x;
	int globalTid = blockIdx.x * blockDim.x + threadIdx.x;

	//Create private histogram for every block (in shared memory) 	
	__shared__ int tempHist[HISTOGRAM_SIZE]; 

	//Put zero's in every private histogram
	tempHist[tid] = 0;
	//Put zero's in global histogram (global memory)
	if (blockId == 0){
		globalHist[tid] = 0;
	}
	__syncthreads();

	
	//Every block calculates his histogram
	if (globalTid < size)
	{
		atomicAdd(&tempHist[arr[globalTid]], 1);
	}
	__syncthreads();
	
	//Add all private histogram to global histogram
	atomicAdd(&globalHist[tid], tempHist[tid]);
}

int* calculateHistogram(int *arr, unsigned int size)
{
	int *dev_arr = 0;
	int *dev_histogram = 0;
	int num_block = size/HISTOGRAM_SIZE +1;
	int* histogram = (int*)calloc(HISTOGRAM_SIZE,sizeof(int)); 
	cudaError_t cudaStatus;

	// Choose which GPU to run on
	cudaStatus = cudaSetDevice(0);
	
	//Allocate memory on device 
	cudaStatus = cudaMalloc((void**)&dev_arr, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc dev_arr failed.");
		return NULL;
	}
	cudaStatus = cudaMalloc((void**)&dev_histogram, HISTOGRAM_SIZE * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc dev_histogram failed.");
		return NULL;
	}

	//Copy initial arr to device (CPU to GPU)
	cudaStatus = cudaMemcpy(dev_arr, arr, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy failed.");
		return NULL;
	}
	//Launch a kernel on the GPU with 256 threads for every block
	addCalculateKernel <<<num_block, HISTOGRAM_SIZE>>>(dev_arr, dev_histogram, size);

	//Copy result to histogram buffer (from GPU to CPU)
	cudaStatus = cudaMemcpy(histogram, dev_histogram, HISTOGRAM_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy failed.");
		return NULL;
	}
	
	cudaFree(dev_arr);
	cudaFree(dev_histogram);

	return histogram;
}
