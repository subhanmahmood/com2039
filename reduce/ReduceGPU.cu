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
 #define BLOCK_SIZE 512
 #define N 1048576
 
 //kernel signature
 __global__ void reduceKernel(float *d_out, float *d_in);
 
 int main(void) {
	 //array size
	 int arraySize = 1 << 18;
	 printf("%i\n", arraySize);
	 //initialise host and device input and output arrays
	 size_t size = arraySize * sizeof(float);
	 size_t size_o = size / BLOCK_SIZE;
 
	 float h_in[arraySize];
	 float h_out[arraySize / BLOCK_SIZE];
	 float *d_in, *d_out;
	

	 //initialise cuda events
	 cudaEvent_t start, stop;
	 cudaEventCreate(&start);
	 cudaEventCreate(&stop);
	 cudaError_t err;
	
	 //fill input array 
	 for (int i = 0; i < arraySize; i++) {
		 h_in[i] = 1.0f;
	 }
	 
	 //allocate space on device for input and output
	 //copy host input to device
	 cudaMalloc((void**) &d_in, size);
	 cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
	 cudaMalloc((void**) &d_out, size_o);
	 
	 //calculate grid size
	 int grid_size = arraySize / BLOCK_SIZE;
	 printf("Grid Size is: %d\n", grid_size);
	 printf("Block Size is: %d\n", BLOCK_SIZE);
 
	 dim3 threadsPerBlock(BLOCK_SIZE);
	 dim3 blocks(grid_size);
	 
	 //start event record
	 cudaEventRecord(start);
	 
	 //call reduce kernel
	 reduceKernel<<<blocks, threadsPerBlock>>>(d_out, d_in);
 
	 cudaDeviceSynchronize();
	 
	 //call reduce kernel with one block and d_out as input
	 reduceKernel<<<1, threadsPerBlock>>>(d_out, d_out);
 
	 // Wait for GPU to finish before accessing on host
	 err = cudaDeviceSynchronize();
	 // err = cudaThreadSynchronize();
	 cudaEventRecord(stop);
 
	 printf("Run kernel: %s\n", cudaGetErrorString(err));
	 
	 //copy device output to host
	 err = cudaMemcpy(h_out, d_out, size_o, cudaMemcpyDeviceToHost);
	 printf("Copy h_out off device: %s\n", cudaGetErrorString(err));
 
	 printf("\n");
 
	 float milliseconds = 0;
	 cudaEventElapsedTime(&milliseconds, start, stop);
	 printf("Elapsed time was: %f milliseconds\n", milliseconds);
 
	 printf("And the final reduction is: %f\n", h_out[0]);
	 
	 //free device memorys
	 cudaFree(d_in);
	 cudaFree(d_out);
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
 