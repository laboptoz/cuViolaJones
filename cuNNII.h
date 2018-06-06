#include <cuda_runtime.h>

__device__ void rowScan(float * input, unsigned int width);

__global__ void downsampleAndRow(unsigned char * input, float * output, unsigned int width, unsigned int scale) {
	unsigned int id = threadIdx.x;
	extern __shared__ float smem[];

	if(scale < 1){
		printf("Error: scale must be 1 =< scale");
	}else if (scale == 1){
		//MOVE TO SHARED MEMORY
	}else{
		smem[id] = (float)input[id*scale + blockIdx.x*width*scale];
		__syncthreads();
		rowScan(smem, width / scale);
		smem[id] += (float)input[id*scale + blockIdx.x*width*scale];
		output[id + blockIdx.x*(width / scale)] = smem[id];
	}
}

__device__ void rowScan(float * input, unsigned int width) {
	unsigned int idx = threadIdx.x;
	int offset = 1;
	//upsweep

	for (int d = width >> 1; d > 0; d >>= 1) {
		__syncthreads();
		if (idx < d) {
			int ai = offset * (2 * idx + 1) - 1;
			int bi = offset * (2 * idx + 2) - 1;
			input[bi] += input[ai];
		}
		offset *= 2;
	}

	//Downsweep
	if (idx == 0) input[width - 1] = 0;
	for (int d = 1; d < width; d *= 2) {
		offset >>= 1;
		__syncthreads();
		if (idx < d) {
			int ai = offset * (2 * idx + 1) - 1;
			int bi = offset * (2 * idx + 2) - 1;

			float t = input[ai];
			input[ai] = input[bi];
			input[bi] += t;
		}
	}
}

__global__ void colScan(float * input, float * output, unsigned int height, unsigned int width) {
	unsigned int idx = threadIdx.x;
	extern __shared__ float smem[];
	smem[idx] = input[idx*width + blockIdx.x];
	__syncthreads();
	int offset = 1;
	//upsweep

	for (int d = height >> 1; d > 0; d >>= 1) {
		__syncthreads();
		if (idx < d) {
			int ai = offset * (2 * idx + 1) - 1;
			int bi = offset * (2 * idx + 2) - 1;
			smem[bi] += smem[ai];
		}
		offset *= 2;
	}

	//Downsweep
	if (idx == 0) smem[height - 1] = 0;
	for (int d = 1; d < height; d *= 2) {
		offset >>= 1;
		__syncthreads();
		if (idx < d) {
			int ai = offset * (2 * idx + 1) - 1;
			int bi = offset * (2 * idx + 2) - 1;

			float t = smem[ai];
			smem[ai] = smem[bi];
			smem[bi] += t;
		}
	}

	__syncthreads();
	//smem[idx] += input[idx*width + blockIdx.x];
	output[idx*width + blockIdx.x] = smem[idx];

}