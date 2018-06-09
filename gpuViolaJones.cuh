#include <cuda_runtime.h>
#include "cuNNII.h"
#include "Filter.cuh"
#include "cuda_error_check.h"
#include "parameter_loader.h"
#include "Point.h"

__global__ void cascadeClassifier(Stage *, unsigned int, unsigned char *, float *, float *, unsigned int, unsigned int, unsigned int, unsigned int);

unsigned char * gpuViolaJones(unsigned char * img, unsigned int width, unsigned int height, unsigned int win_size, float scale) {
	
	printf("Loading cascade classifier information to GPU\n");
	//READ IN CASCADE CLASSIFIER INFORMATION
	unsigned int * num_stages = new unsigned int;
	Stage * stages_gpu = loadParametersToGPU(num_stages);

	printf("Generating image and squared image pyramids\n");
	//GENERATE THE INTEGRAL IMAGE PYRAMID
	unsigned int * iiPyramidSizes = nullptr;
	unsigned int iiPyramidDepth = 0;
	float ** iiPyramid_gpu = generateImagePyramid<false>(img, &iiPyramidSizes, &iiPyramidDepth, win_size, width, height, scale);

	//GENERATE THE SQUARED INTEGRAL IMAGE PYRAMID
	unsigned int * viiPyramidSizes = nullptr;
	unsigned int viiPyramidDepth = 0;
	float ** viiPyramid_gpu = generateImagePyramid<true>(img, &viiPyramidSizes, &viiPyramidDepth, win_size, width, height, scale);
	
	unsigned char * activationMask = (unsigned char *)malloc(sizeof(unsigned char)*iiPyramidSizes[0] * iiPyramidSizes[1]);
	unsigned char * activationMask_gpu;
	CHECK(cudaMalloc(&activationMask_gpu, sizeof(unsigned char)*iiPyramidSizes[0] * iiPyramidSizes[1]));

	//Loop through depth levels
	for (unsigned int i = 0; i < 1; i++) {
		printf("Depth: %u, Width: %u, Height: %u, Window Size: %u\n", i, iiPyramidSizes[i * 2], iiPyramidSizes[2 * i + 1], win_size);
		unsigned int threads = max(iiPyramidSizes[i * 2]-win_size, 1024);
		dim3 block = dim3(threads);
		dim3 grid = dim3((iiPyramidSizes[2*i]-win_size+1023)/1024, iiPyramidSizes[2*i+1]- win_size);

		cascadeClassifier<<<grid,block>>>(stages_gpu, *num_stages, activationMask_gpu, iiPyramid_gpu[i], viiPyramid_gpu[i], i, win_size, iiPyramidSizes[i * 2], iiPyramidSizes[i * 2 + 1]);
		CHECK(cudaDeviceSynchronize());

	}

	CHECK(cudaMemcpy(activationMask, activationMask_gpu, sizeof(unsigned char)*iiPyramidSizes[0] * iiPyramidSizes[1], cudaMemcpyDeviceToHost));

	return activationMask;
}
__global__ void cascadeClassifier(Stage * stages, unsigned int num_stages, unsigned char * activationMask, float * iiPyramid, float * viiPyramid, unsigned int depth, unsigned int window_size, unsigned int width, unsigned int height) {
	unsigned int offset = threadIdx.x + blockIdx.x*blockDim.x + blockIdx.y*width;
	float mean = 0;
	float norm_factor = 0;
	unsigned int pass = 0;
	norm_factor = viiPyramid[blockIdx.y*width + threadIdx.x]
				- viiPyramid[blockIdx.y*width + threadIdx.x + window_size]
				- viiPyramid[blockIdx.y*width + threadIdx.x + window_size]
				+ viiPyramid[blockIdx.y*width + threadIdx.x + window_size + window_size*width];
	mean	= iiPyramid[blockIdx.y*width + threadIdx.x]
			- iiPyramid[blockIdx.y*width + threadIdx.x + window_size]
			- iiPyramid[blockIdx.y*width + threadIdx.x + window_size]
			+ iiPyramid[blockIdx.y*width + threadIdx.x + window_size + window_size*width];
	norm_factor /= (window_size);
	norm_factor = norm_factor - mean*mean;
	norm_factor = (norm_factor > 0) * sqrtf(norm_factor) + (norm_factor <= 0);
	for (int i = 0; i < num_stages; i++) {
		pass += 1*stages[i].getValue<float>(iiPyramid + offset, norm_factor, width);
	}
	activationMask[blockIdx.y*width + threadIdx.x] = (unsigned char) (255*((1.0*pass)/num_stages));
}