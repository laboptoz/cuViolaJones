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
	int height = 20;
	int width = 10;
	int scale = 2;
	if (width / scale % 2 != 0) {
		width -= scale;
	}
	if (height / scale % 2 != 0) {
		height -= scale;
	}
	unsigned char * input = new unsigned char[height*width];
	for (int i = 0; i < height*width; i++) {
		input[i] = i+1;
	}

	unsigned char * img_gpu;
	float * ii_gpu;
	float * ii_cpu = (float *)malloc(sizeof(float) * (width/scale)*(height/scale));
	CHECK(cudaMalloc(&img_gpu, sizeof(unsigned char)*height*width));
	CHECK(cudaMalloc(&ii_gpu, sizeof(float) * (width / scale)*(height / scale)));
	CHECK(cudaMemcpy(img_gpu, input, sizeof(unsigned char)*height*width, cudaMemcpyHostToDevice));
	downsampleAndRow<<<height / scale, width / scale, width/scale*sizeof(float)>>> (img_gpu, ii_gpu, width, scale);
	CHECK(cudaDeviceSynchronize());

	colScan <<<width / scale, height / scale, height / scale * sizeof(float) >> > (ii_gpu, ii_gpu, height / scale, width / scale);
	CHECK(cudaDeviceSynchronize());

	CHECK(cudaMemcpy(ii_cpu, ii_gpu, sizeof(float) * (width / scale) * (height / scale), cudaMemcpyDeviceToHost));
	for (int i = 0; i < height/scale; i++) {
		for (int j = 0; j < width/scale; j++) {
			printf("%f, ", ii_cpu[i*width/scale+j]);
		}
		printf("\n");
	}
}

__global__ void cuViolaJones(unsigned char * img, unsigned int height, unsigned int width) {

}