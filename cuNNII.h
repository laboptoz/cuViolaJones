#include <cuda_runtime.h>

__device__ void rowScan(float * input, unsigned int width);
__inline__ __device__ unsigned int smallPow2(unsigned int width);

__global__ void downsampleAndRow(unsigned char * input, float * output, unsigned int width, float scale) {
	unsigned int idx = threadIdx.x;
	unsigned int row = blockIdx.y;
	unsigned int row_scaled = (unsigned int)(round(scale*blockIdx.y));
	unsigned int width_scaled = round(width / scale);
	unsigned int idx_scaled = (unsigned int)(round(scale*idx) + blockIdx.x*1024);

	extern __shared__ float smem[];
	smem[idx] = (float) input[idx_scaled + row_scaled*width];
	smem[width_scaled + idx] = smem[idx];
	rowScan(smem, width_scaled);
	output[idx+blockIdx.y*width_scaled] = smem[width_scaled + idx];
}

__device__ void rowScan(float * input, unsigned int width) {
	unsigned int idx = threadIdx.x;
	unsigned int width_offset = 0;
	unsigned int remainder = 0;
	while (width_offset < width) {
		unsigned int offset = 1;
		unsigned int pw2 = smallPow2(width-width_offset);
		unsigned int working_rem = remainder;
		if (pw2 != 1) {
			for (int i = pw2 / 2; i > 0; i /= 2) {
				__syncthreads();
				if (idx < i) {
					input[width + offset * (2 * idx + 2) - 1 + width_offset] += input[width + offset * (2 * idx + 1) - 1 + width_offset];
				}
				offset *= 2;
			}
			if (idx == 0) {
				remainder += input[width + pw2 + width_offset - 1];
				input[width + pw2 + width_offset - 1] = 0;
			}
			for (int i = 1; i < pw2; i *= 2) {
				offset /= 2;
				__syncthreads();
				if (idx < i) {
					float swap = input[width + offset*(2 * idx + 1) - 1 + width_offset];
					input[width + offset*(2 * idx + 1) - 1 + width_offset] = input[width + offset*(2 * idx + 2) - 1 + width_offset];
					input[width + offset*(2 * idx + 2) - 1 + width_offset] = swap;
				}
			}
			if (idx < pw2) {
				input[width + idx + width_offset] += input[idx + width_offset] + working_rem;
			}
		} else {
			if (idx < pw2) {
				input[width + idx + width_offset] = input[idx + width_offset] + working_rem;
			}
		}
		__syncthreads();
		width_offset += pw2;
	}
	
}

__global__ void colScan(float * in_arr, unsigned int height, unsigned int width) {
	extern __shared__ float input[];
	unsigned int idx = threadIdx.x;
	unsigned int width_offset = 0;
	unsigned int remainder = 0;
	while (width_offset < width) {
		unsigned int offset = 1;
		unsigned int pw2 = smallPow2(width - width_offset);
		unsigned int working_rem = remainder;
		if (pw2 != 1) {
			for (int i = pw2 / 2; i > 0; i /= 2) {
				__syncthreads();
				if (idx < i) {
					input[width + offset * (2 * idx + 2) - 1 + width_offset] += input[width + offset * (2 * idx + 1) - 1 + width_offset];
				}
				offset *= 2;
			}
			if (idx == 0) {
				remainder += input[width + pw2 + width_offset - 1];
				input[width + pw2 + width_offset - 1] = 0;
			}
			for (int i = 1; i < pw2; i *= 2) {
				offset /= 2;
				__syncthreads();
				if (idx < i) {
					float swap = input[width + offset*(2 * idx + 1) - 1 + width_offset];
					input[width + offset*(2 * idx + 1) - 1 + width_offset] = input[width + offset*(2 * idx + 2) - 1 + width_offset];
					input[width + offset*(2 * idx + 2) - 1 + width_offset] = swap;
				}
			}
			if (idx < pw2) {
				input[width + idx + width_offset] += input[idx + width_offset] + working_rem;
			}
		}
		else {
			if (idx < pw2) {
				input[width + idx + width_offset] = input[idx + width_offset] + working_rem;
			}
		}
		__syncthreads();
		width_offset += pw2;
	}
}

__inline__ __device__ unsigned int smallPow2(unsigned int width) {
	width = width | (width >> 1);
	width = width | (width >> 2);
	width = width | (width >> 4);
	width = width | (width >> 8);
	width = width | (width >> 16);
	return width - (width >> 1);
}