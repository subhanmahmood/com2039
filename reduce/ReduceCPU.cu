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
#define BLOCK_SIZE 32
#define N 1024

using namespace std;

//kernel signature
__global__ void reduceKernel(float *d_out, float *d_in);


//function to call kernel and return csv string
string reduceInvoker(int arraySize) {
		//initialise timing variable for host code
		clock_t h_start, h_end;

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

		// Wait for GPU to finish before accessing on host
		err = cudaDeviceSynchronize();
		//finish event record
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

		//start host timer
		h_start = clock();
		//calculate final reduction on host
		float final_reduction = 0.0f;
		for (int i = 0; i < grid_size; i++) {
			final_reduction += h_out[i];
		}
		//stop host timer
		h_end = clock();

		//calculate totat elapsed time
		float h_milliseconds = float(h_end - h_start) / float(CLOCKS_PER_SEC);
		milliseconds += h_milliseconds;
		printf("Elapsed time was: %f milliseconds\n", milliseconds);
		printf("And the final reduction is: %f\n", final_reduction);

		//free device memory
		cudaFree(d_in);
		cudaFree(d_out);

		//return csv string
		string csv = to_string(arraySize) + "," + to_string(milliseconds) +  "," + to_string(BLOCK_SIZE) + "," + to_string(grid_size) + "\n";
        return csv;
}

int main(void) {
	//Open file
	std::ofstream myFile;
	myFile.open("times-cpu.csv", std::ofstream::trunc);
	//Write headers
	myFile << "Array Size,Elapsed Time,Block Size,Grid Size\n";

	//Call reduce function invoker for different array sizes
	for(unsigned int i = 1; i <= (1 << 20) + 3; i <<= 1){
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
