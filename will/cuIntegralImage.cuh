#include <cuda_runtime.h>

__device__ void scan(unsigned char * img, float * output, unsigned int width) {
	unsigned int idx = threadIdx.x;
	unsigned int id = threadIdx.x + blockIdx.x*width;
	extern __shared__ float temp[];
	int offset = 1;

	//upsweep
	temp[2 * idx] = (float)img[2 * idx + blockIdx.x*width];
	temp[2 * idx+1] = (float)img[2 * idx + blockIdx.x*width + 1];

	for (int d = width >> 1; d > 0; d >>= 1) {
		__syncthreads();
		if (idx < d) {
			int ai = offset * (2 * idx + 1) - 1;
			int bi = offset * (2 * idx + 2) - 1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	//Downsweep
	if (idx == 0) temp[width - 1] = 0;
	for (int d = 1; d < width; d *= 2) {
		offset >>= 1;
		__syncthreads();
		if (idx < d) {
			int ai = offset * (2 * idx + 1) - 1;
			int bi = offset * (2 * idx + 2) - 1;

			float t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}

	__syncthreads();
	output[2 * idx + blockIdx.x*width] = temp[2 * idx];
	output[2 * idx + 1 + blockIdx.x*width] = temp[2 * idx + 1];
}

__global__ void cuIntegralImage(unsigned char * img, float * ii, unsigned int height, unsigned int width) {
	scan(img, ii, width);
}