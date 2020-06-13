#include <stdio.h>
#include <numeric>
#include <stdlib.h>
#include <cuda.h>
#include <iostream>
#include <fstream>
#include <math.h>
#define BLOCK_SIZE 256

using namespace std;

//Kernel method signatures
__global__ void simple_histogram(int *d_bins, const int *d_in, int BIN_COUNT);

__global__ void shmem_histogram(int *d_bins, const int *d_in, int BIN_COUNT,
		int N);

//Function to call kernel and return csv string
string callKernel(int arraySize) {

	//initialise cuda timing events
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	int N = arraySize;
	// This is the size of the input array
	int *d_bins;
	// This is the array that will contain the histogram
	int *d_in;
	// This is the array that will contain the input data
	int BIN_COUNT = 8; // This sets the number of bins in the histogram
	// We will use the CUDA unified memory model to ensure data is transferred between host and device
	cudaMallocManaged(&d_bins, BIN_COUNT * sizeof(int));
	cudaMallocManaged(&d_in, N * sizeof(int));
	// Now we need to generate some input data
	// You can invent any strategy you like for generating the input data!
	for (int i = 0; i < N; i++) {
		d_in[i] = i;
	}
	// We also need to initialise the bins in the histogram
	for (int i = 0; i < BIN_COUNT; i++) {
		d_bins[i] = 0;
	}
	// Now we need to set up the grid size. Work on the assumption that N is an exact multiple
	// of BLOCK_SIZE
	// for the moment
	int grid_size = (int) ceil(double(N) / BLOCK_SIZE);
	printf("Array Size: %i\n", N);
	printf("Grid Size: %i\n", grid_size);
	printf("Block Size: %i\n\n", BLOCK_SIZE);
	cudaEventRecord(start);

	//Shared memory kernel call
	shmem_histogram<<<grid_size, BLOCK_SIZE, BIN_COUNT>>>(d_bins, d_in, BIN_COUNT, N);

	//Unified memory kernel call
	//simple_histogram<<<grid_size, BLOCK_SIZE>>>(d_bins, d_in, BIN_COUNT);

	// wait for Device to finish before accessing data on the host
	cudaDeviceSynchronize();
	cudaEventRecord(stop);

	// Now we can print out the resulting histogram
	int tmp = 0;
	for (int i = 0; i < BIN_COUNT; i++) {
		tmp += d_bins[i];
		printf("Bin no. %d: Count = %d\n", i, d_bins[i]);
	}
	
	//calculate elapsed time
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("\nElapsed time was: %f milliseconds\n", milliseconds);
	printf("Total elements: %i\n\n", tmp);

	//return csv string
	string csv = to_string(arraySize) + "," + to_string(milliseconds) + ","
			+ to_string(BLOCK_SIZE) + "," + to_string(grid_size) + "\n";
	return csv;

}

int main(void) {
	std::ofstream myFile;
	myFile.open("times.csv", std::ofstream::trunc);
	//Write headers
	myFile << "Array Size,Elapsed Time,Block Size,Grid Size\n";

	//Call histogram function invoker for different array sizes
	for (unsigned int i = BLOCK_SIZE; i <= (1 << 20) + 3; i <<= 1) {
		printf("%i\n", i);
		string csv = callKernel(i);
		//write to file
		myFile << csv;
	}

	myFile.close();
	return 0;
}

//global memory kernel
__global__ void simple_histogram(int *d_bins, const int *d_in,
		const int BIN_COUNT) {
	//global thread id
	int myId = threadIdx.x + blockDim.x * blockIdx.x;
	//get item from input array
	int myItem = d_in[myId];
	//calculate the bin that the item will go in
	int myBin = myItem % BIN_COUNT;
	//use atomic add to increment the bin's value in global memory
	//atomic add means threads must wait for other threads to finish 
	//updating the value before they attempt to themselves
	atomicAdd(&(d_bins[myBin]), 1);
}

//shared memory kernel
__global__ void shmem_histogram(int *d_bins, const int *d_in, int BIN_COUNT,
		int N) {
	//global thread id
	int thIdx = threadIdx.x + blockDim.x * blockIdx.x;
	//get item from input array
	int myItem = d_in[thIdx];

	//create shared memory array
	//extern used as BIN_COUNT is not a constant value
	extern __shared__ int s_bins[];

	//check that local thread id is less than the bin count
	if (threadIdx.x < BIN_COUNT) {
		s_bins[threadIdx.x] = 0;
	}


	//make sure that all threads have finished executing before continuing to prevent conflicts
	__syncthreads();

	//check that global thread id is less than input array size
	if (thIdx < N) {
		//calculate bin
		int bin = myItem % BIN_COUNT;
		//increment bin value in shared memory
		atomicAdd(&s_bins[bin], 1);
	}

	//make sure all threads have finished executing before continuing
	__syncthreads();

	if (threadIdx.x < BIN_COUNT) {
		//use atomic add to increment global bin value
		//quicker as less calls to global memory
		atomicAdd(&d_bins[threadIdx.x], s_bins[threadIdx.x]);
	}
}