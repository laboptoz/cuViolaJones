#include <cuda_runtime.h>
//#include "cuIntegralImage.cuh"
#include "cuNNII.h"

#define CHECK(ans) { err_check((ans), __FILE__, __LINE__); }
inline void err_check(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "Error: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


void gpuViolaJones() {
	//MAX OF 1024 RIGHT NOW
	int height = 2048;
	int width = 2048;
	float scale = 1.2;

	unsigned char * input = new unsigned char[height*width];
	for (int i = 0; i < height*width; i++) {
		input[i] = i+1;
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

	downsampleAndRow<<<grid, block, 2* scaled_width *sizeof(float)>>> (img_gpu, ii_gpu, width, scale);
	CHECK(cudaDeviceSynchronize());

	dim3 block_colscan(scaled_height);
	dim3 grid_colscan((height + 1023) / 1024, scaled_width);
	colScan << <grid_colscan, block_colscan, 2 * scaled_height * sizeof(float) >> > (ii_gpu, scaled_height, scaled_width);
	CHECK(cudaDeviceSynchronize());
	
	CHECK(cudaMemcpy(ii_cpu, ii_gpu, sizeof(float) * scaled_width * scaled_height, cudaMemcpyDeviceToHost));
	
	/*for (int i = 0; i < scaled_height; i++) {
		for (int j = 0; j < scaled_width; j++) {
			printf("%f\t", ii_cpu[i*scaled_width + j]);
		}
		printf("\n");
	}*/
}

__global__ void cuViolaJones(unsigned char * img, unsigned int height, unsigned int width) {

}