#include <stdio.h>
#include <numeric>
#include <stdlib.h>
#include <cuda.h>
#include <iostream>
#include <fstream>

using namespace std;

#define BLOCK_SIZE 1024

__global__ void reduceKernel(float *device_output, float *device_input, float *final_output);

//function to call reduce kernel for array size
string reduceInvoker(int arraySize) {
	int N = arraySize;
	printf("%i\n", N);

	//Create variables for input and output array sizes
	size_t size = N * sizeof(float);
	size_t size_output = size / BLOCK_SIZE;
	size_t size_final_output = sizeof(float);

	//initialising host input and output arrays
	float host_input[N];
	float host_output[1];

	//initialising pointers for device arrays
	float *device_input, *device_output, *final_output;

	//initialising variables and events for cuda timer
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaError_t err;

	//filling host input array with values
	for (int i = 0; i < N; i++) {
		host_input[i] = 1.0f;
	}

	//allocating space for device input array
	cudaMalloc((void**) &device_input, size);
	//copying host input array to device
	cudaMemcpy(device_input, host_input, size, cudaMemcpyHostToDevice);
	//allocating space for device and final output arrays
	cudaMalloc((void**) &device_output, size_output);
	cudaMalloc((void**) &final_output, size_final_output);

	//calculating grid size
	int grid_size = N / BLOCK_SIZE;
	printf("Grid Size is: %d\n", grid_size);
	printf("Block Size is: %d\n", BLOCK_SIZE);

	dim3 threadsPerBlock(BLOCK_SIZE);
	dim3 blocks(grid_size);
	//start timer
	cudaEventRecord(start);
	//call kernel
	reduceKernel<<<blocks, threadsPerBlock>>>(device_output, device_input, final_output);


	//stop timer
	cudaEventRecord(stop);
	// Wait for GPU to finish before accessing on host
	err = cudaDeviceSynchronize();
	printf("Run kernel: %s\n", cudaGetErrorString(err));
	printf("\n");
	//copy final output array to host
	err = cudaMemcpy(host_output, final_output, size_final_output, cudaMemcpyDeviceToHost);
	printf("Copy host_output off device: %s\n", cudaGetErrorString(err));
	printf("\n");

	//calculate elapsed time
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Elapsed time was: %f\n", milliseconds);

	printf("And the final reduction is: %f\n", host_output[0]);

	//free device memory
	cudaFree(device_input);
	cudaFree(device_output);
	cudaFree(final_output);
	//format string for csv file
	string csv = to_string(arraySize) + "," + to_string(milliseconds) + ","
			+ to_string(BLOCK_SIZE) + "," + to_string(grid_size) + "\n";
	return csv;
}

int main(void) {
	//Open file
	std::ofstream myFile;
	myFile.open("times.csv", std::ofstream::trunc);
	//Write headers
	myFile << "Array Size,Elapsed Time,Block Size,Grid Size\n";

	//Call reduce function invoker for different array sizes and write result to file
	for (unsigned int i = 1; i <= (1 << 20); i <<= 1) {
		printf("%i\n", i);
		string csv = reduceInvoker(i);
		myFile << csv;
	}

	//close file
	myFile.close();
}
__global__ void reduceKernel(float* device_output, float* device_input, float *final_output) {
	//work out global thread id
	int myId = threadIdx.x + blockDim.x * blockIdx.x; // ID relative
	//work out local thread id
	int tid = threadIdx.x; // Local ID
	//initialise static shared memory
	__shared__ float temp[BLOCK_SIZE];
	//set value in shared memory
	temp[tid] = device_input[myId];
	//make sure threads are synced before adding values
	__syncthreads();
	// do reduction in shared memory
	for (unsigned int s = blockDim.x / 2; s >= 1; s >>= 1) {
		if (tid < s) {
			temp[tid] += temp[tid + s];
		}
		__syncthreads(); // make sure all adds at one stage are
	}
	// only thread 0 writes result for this block back to global memory
	if (tid == 0) {
		device_output[blockIdx.x] = temp[tid];
		//add partial reductions to get final reduction
		atomicAdd(&final_output[0], device_output[blockIdx.x]);
	}
}
