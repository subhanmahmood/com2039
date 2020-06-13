#include <stdio.h>
#include <numeric>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>

#define N 64
#define BLOCK_SIZE 16

//kernel method signatures
__global__ void scanKernel(int n, float *input_data, float *last_elements);

__global__ void scanKernel(int n, float *input_data);

__global__ void addScanKernel(float *input_data, float *last_elements);

int main(void) {

	//initialise array for input array and array of last elements of each block from first scan
	float *input_data;
	float *last_elements;

	//initialise cuda timing events
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaError_t err;

	printf("Block size: %i\n", BLOCK_SIZE);
	printf("Array size: %i\n", N);

	//calculate grid size
	int grid_size = (int) ceil((double(N) / BLOCK_SIZE));
	printf("Grid size is %i\n", grid_size);

	// Allocate Unified Memory - accessible from CPU or GPU
	cudaMallocManaged(&input_data, N * sizeof(float));
	cudaMallocManaged(&last_elements, N * sizeof(float));

	// Initialise the input data on the host
	// Making it easy to test the result
	for (int i = 0; i < N; i++) {
		input_data[i] = 1.0f;
	}
	//First scan
	// Run the kernel

	//start event record
	cudaEventRecord(start);

	//call scan kernel on input array to perform scan on input
	//last elements from each block will be written to last_elements
	scanKernel<<<grid_size, BLOCK_SIZE>>>(BLOCK_SIZE, input_data,
			last_elements);

	//wait for device to finish before continuing
	cudaDeviceSynchronize();

	//print resulting array
	printf("resulting array:\n");
	for (int i = 0; i < N; i++) {
		printf("%f ", input_data[i]);
		if ((i + 1) % (BLOCK_SIZE) == 0) {
			printf("\n");
		}
	}

	//calculate new grid size for second kernel call
	int new_grid_size = (int) ceil(double(grid_size) / BLOCK_SIZE);
	printf("New grid size is %i\n", new_grid_size);

	//call scan kernel on array of last elements
	scanKernel<<<new_grid_size, BLOCK_SIZE>>>(BLOCK_SIZE, last_elements);
	cudaDeviceSynchronize();

	printf("\n");

	//call addScanKernel to add elements from last_elements to input_data
	addScanKernel<<<grid_size, BLOCK_SIZE>>>(input_data, last_elements);

	//stop event record
	cudaEventRecord(stop);

	//wait for device to finish before continuing
	cudaDeviceSynchronize();
	//calculate elapsed time
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Elapsed time was: %fms\n", milliseconds);
	
	//print final value
	printf("final value: %f", input_data[N - 1]);

	cudaFree(input_data);
	cudaFree(last_elements);
	return 0;
}

__device__ void scanDevice(int n, float *idata, int thIdx, int tid) {

	// We need to create two buffers to avoid the race condition - read from one and write into the other
	__shared__ float temp[BLOCK_SIZE];
	__shared__ float temp2[BLOCK_SIZE];
	// declare a boolean to identify which of the two buffers we are currently reading from
	bool tempSelector = true;
	// each thread reads one data item into the first buffer in shared memory
	temp[tid] = idata[thIdx];
	__syncthreads();
	for (int offset = 1; offset < n; offset *= 2) {
		// Let's also make sure that threads are only working on array elements that have data
		if (tid >= offset && thIdx < N) {
			if (tempSelector) // for odd loop numbers, read from first buffer into second
			{
				temp2[tid] = temp[tid] + temp[tid - offset];
			} else // for even loop numbers, read from second buffer into first
			{
				temp[tid] = temp2[tid] + temp2[tid - offset];
			}
		}
		// We also need to make sure all the unmodified values are copied between the two buffers
		if (tid < offset) {
			if (tempSelector)
				temp2[tid] = temp[tid];
			else
				temp[tid] = temp2[tid];
		}
		// and update the condition
		tempSelector = !tempSelector;
		// then make sure all threads have finished before going round the loop again
		__syncthreads();

	}
	// we need to make sure we output the value from the correct buffer
	if (tempSelector) {
		idata[thIdx] = temp[tid];
	} else {
		idata[thIdx] = temp2[tid];
	}
}

__global__ void scanKernel(int n, float *input_data) {
	//calculate global thread id
	int thIdx = threadIdx.x + blockIdx.x * blockDim.x;
	scanDevice(n, input_data, thIdx, threadIdx.x);
}

__global__ void scanKernel(int n, float *input_data, float *last_elements) {
	//calculate global thread id
	int thIdx = threadIdx.x + blockIdx.x * blockDim.x;
	scanDevice(n, input_data, thIdx, threadIdx.x);

	//check if thread is last thread in block
	if (thIdx == (blockIdx.x + 1) * blockDim.x - 1) {
		//printf("end of block, tid:%i, blockid: %i. idata: %f\n", thIdx, blockIdx.x, idata[thIdx]);

		//write value for current thread to last_elements at the current block index
		last_elements[blockIdx.x] = input_data[thIdx];
	}

}

__global__ void addScanKernel(float *input_data, float *last_elements) {
	//global id
	int thIdx = threadIdx.x + blockIdx.x * blockDim.x;

	//if not in first block
	if (blockIdx.x != 0) {
		//increment value at global thread id in input array by the corresponding element in last_elements
		input_data[thIdx] += last_elements[blockIdx.x - 1];
	}

	//printf("Thread:  %i, First scan: %i, Last elem: %i\n", thIdx, globalval, addval);
}

