#include <cuda_runtime.h>
//#include "cuIntegralImage.cuh"
#include "cuNNII.h"
#include "Filter.cuh"
#include "cuda_error_check.h"
#define TEST 1

//TESTING FUNCTION
__global__ void testfunc(float * input, unsigned int width);
//void cuNNII_example(int height, int width, float scale);

void gpuViolaJones() {
	unsigned char * input = new unsigned char[8*8];
	for (int i = 0; i < 8 * 8; i++) {
		input[i] = i + 1;
	}
	unsigned int * pyramidSizes = nullptr;
	float ** gpuPyramid = generateImagePyramid(input, &pyramidSizes, 2, 8, 8, 1.2);
	printf("Testing\n");
	testfunc <<<1,1>>>(gpuPyramid[0], pyramidSizes[0]);

}


__global__ void testfunc(float * input, unsigned int width) {
	Filter f = Filter(0, 0, 2, 2, -1, 0, 2, 2, 2, 1, 2, 0, 3);
	float num =  f.getValue(input, width);
	printf("\n\n%f\n\n", num);
}

/*void cuNNII_example(int height, int width, float scale) {
	unsigned char * input = new unsigned char[height*width];
	for (int i = 0; i < height*width; i++) {
		input[i] = i + 1;
	}
	unsigned int scaled_height = round(height / scale);
	unsigned int scaled_width = round(width / scale);
	unsigned char * img_gpu;
	float * ii_gpu;
	float * ii_cpu = (float *)malloc(sizeof(float) * scaled_height * scaled_width);

	CHECK(cudaMalloc(&img_gpu, sizeof(unsigned char)*height*width));
	CHECK(cudaMalloc(&ii_gpu, sizeof(float)  * scaled_height * scaled_width));

	CHECK(cudaMemcpy(img_gpu, input, sizeof(unsigned char)*height*width, cudaMemcpyHostToDevice));

	dim3 block((unsigned int)scaled_width);
	dim3 grid((width + 1023) / 1024, scaled_height);

	downsampleAndRow << <grid, block, 2 * scaled_width * sizeof(float) >> > (img_gpu, ii_gpu, width, scale);
	CHECK(cudaDeviceSynchronize());

	dim3 block_colscan(scaled_height);
	dim3 grid_colscan((height + 1023) / 1024, scaled_width);
	colSum << <grid_colscan, block_colscan, 2 * scaled_height * sizeof(float) >> > (ii_gpu, scaled_height, scaled_width);
	CHECK(cudaDeviceSynchronize());

	CHECK(cudaMemcpy(ii_cpu, ii_gpu, sizeof(float) * scaled_width * scaled_height, cudaMemcpyDeviceToHost));

	for (int i = 0; i < scaled_height; i++) {
		for (int j = 0; j < scaled_width; j++) {
			printf("%f\t", ii_cpu[i*scaled_width + j]);
		}
		printf("\n");
	}

	test << <1, 1 >> > (ii_gpu, scaled_width);

	free(ii_cpu);
	cudaFree(ii_gpu);
	cudaFree(img_gpu);
	delete[] input;
}*/