/*
 ============================================================================
 Name        : Lab5.cu
 Author      : sm01800
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>
#include <numeric>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include <fstream>

using namespace std;
#define BLOCK_SIZE 32
#define N 1048576

//function to call kernel and return csv string
__global__ void reduceKernel(float *d_out, float *d_in);

//function to calculate the log (base BLOCK SIZE) of an integer n
//and return rounded up value
int logBaseBlockSize(int n) {
	return (int) ceil((log(n) / log(BLOCK_SIZE)));
}


//function to call kernel and return csv string
string reduceInvoker(int arraySize) {

	//initialise host and device input and output arrays
	size_t size = arraySize * sizeof(float);
	size_t size_o = size / BLOCK_SIZE;

	float h_in[arraySize];
	float h_out[arraySize / BLOCK_SIZE];
	float *d_in, *d_out;

	//initialise cuda timing events for device code
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaError_t err;

	//fill host input array with 1s
	for (int i = 0; i < arraySize; i++) {
		h_in[i] = 1.0f;
	}

	//allocate space for device input and output arrays and copy host input to device
	cudaMalloc((void**) &d_in, size);
	cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**) &d_out, size_o);

	//calculate grid size
	int grid_size = arraySize / BLOCK_SIZE;
	printf("Grid Size is: %d\n", grid_size);
	printf("Block Size is: %d\n", BLOCK_SIZE);

	//set grid and block sizes
	dim3 threadsPerBlock(BLOCK_SIZE);
	dim3 blocks(grid_size);

	//start event record
	cudaEventRecord(start);
	//call kernel
	reduceKernel<<<blocks, threadsPerBlock>>>(d_out, d_in);
	//check that grid size is greater than 0 to prevent illegal memory access
	if (grid_size > 0) {
		//loop from 0 to the logBaseBlockSize of the array size
		//this gives us the number of kernel calls we need to reduce to a single value
		for (int i = 0; i < logBaseBlockSize(N) - 1; i++) {
			//calculate new grid size
			int i_grid_size = grid_size / BLOCK_SIZE;
			dim3 blocks(i_grid_size);
			//call kernel with new grid size and device output array as input
			reduceKernel<<<blocks, threadsPerBlock>>>(d_out, d_out);
			//wait for kernel to finish before continuing
			cudaDeviceSynchronize();

		}
	}

	// Wait for GPU to finish before accessing on host
	err = cudaDeviceSynchronize();
	// err = cudaThreadSynchronize();
	cudaEventRecord(stop);

	printf("Run kernel: %s\n", cudaGetErrorString(err));

	//copy device output array to host
	err = cudaMemcpy(h_out, d_out, size_o, cudaMemcpyDeviceToHost);
	printf("Copy h_out off device: %s\n", cudaGetErrorString(err));
	
	printf("\n");

	//calculate elapsed time of kernel 
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Elapsed time was: %f milliseconds\n", milliseconds);
	printf("And the final reduction is: %f\n", h_out[0]);

	//free device memory
	cudaFree(d_in);
	cudaFree(d_out);

	//return csv string
	string csv = to_string(arraySize) + "," + to_string(milliseconds) + ","
			+ to_string(BLOCK_SIZE) + "," + to_string(grid_size) + "\n";
	return csv;
}

int main(void) {
	//Open file
	std::ofstream myFile;
	myFile.open("times-gpu.csv", std::ofstream::trunc);
	//Write headers
	myFile << "Array Size,Elapsed Time,Block Size,Grid Size\n";

	//Call reduce function invoker for different array sizes
	for (unsigned int i = 1; i <= (1 << 20); i <<= 1) {
		printf("%i\n", i);
		string csv = reduceInvoker(i);
		myFile << csv;
	}

	myFile.close();
}

__global__ void reduceKernel(float* d_out, float* d_in) {
	int myId = threadIdx.x + blockDim.x * blockIdx.x; // ID relative to whole array
	int tid = threadIdx.x; // Local ID	within the	current block
	__shared__ float temp[BLOCK_SIZE];
	temp[tid] = d_in[myId];
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
		d_out[blockIdx.x] = temp[tid];
	}
}
