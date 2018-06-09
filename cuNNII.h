#include <cuda_runtime.h>
#include "cuda_error_check.h"
#include <iostream>
#include <fstream>
#define TEST 1

__device__ void rowSum(float * input, unsigned int width);
__inline__ __device__ unsigned int smallPow2(unsigned int width);
template <class T> __global__ void downsampleAndRow(T * input, float * float_img, float * output, unsigned int width, float scale);
__global__ void colSum(float * arr, unsigned int height, unsigned int width);

template<bool variance>
float ** generateImagePyramid(unsigned char * original, unsigned int ** sizes_ptr, unsigned int * depth, unsigned int min_size, unsigned int width, unsigned int height, float scale) {
	printf("Generating image pyramid on GPU\n");
	//CUDA MALLOC AND COPY THE ORIGINAL IMAGE
	unsigned char * orig_img_gpu;
	CHECK(cudaMalloc(&orig_img_gpu, sizeof(unsigned char)*width*height));
	CHECK(cudaMemcpy(orig_img_gpu, original, sizeof(unsigned char)*width*height, cudaMemcpyHostToDevice));

	float * float_img_gpu;
	CHECK(cudaMalloc(&float_img_gpu, sizeof(float)*width*height));

	//DETERMINE PYRAMID DEPTH
	unsigned int scaled_width = width;
	unsigned int scaled_height = height;
	*depth = 0;
	unsigned int sum_size = 0;

	if (min_size <= 3) {
		min_size = 4;
	}
	while (scaled_width >= min_size && scaled_height >= min_size) {
		(*depth)++;
		sum_size += (scaled_width+1)*(scaled_height+1);
		scaled_width = round(scaled_width / scale);
		scaled_height = round(scaled_height / scale);
	}
	printf("Pyramid depth is: %u\n", *depth);
	//CUDA MALLOC THE IMAGES
	float * imagePyramid_gpu;
	CHECK(cudaMalloc(&imagePyramid_gpu, sum_size * sizeof(float)));

	float ** integralimages_gpu = new float *[*depth];

	//Assign pyramid sizes to argument 'sizes'
	sum_size = 0;
	scaled_width = width;
	scaled_height = height;
	*sizes_ptr = (unsigned int *) malloc(2*(*depth)*sizeof(unsigned int));
	unsigned int * sizes = *sizes_ptr;
	for (int i = 0; i < *depth; i++) {
		integralimages_gpu[i] = imagePyramid_gpu + sum_size;
		sum_size += (scaled_width+1)*(scaled_height+1);
		sizes[2*i] = scaled_width+1;
		sizes[2*i + 1] = scaled_height+1;
		scaled_width = round(scaled_width / scale);
		scaled_height = round(scaled_height / scale);
	}
	
	
	//GENERATE SIZE 1 INTEGRAL IMAGE
	scaled_width = width;
	scaled_height = height;
	
	downsampleAndRow <unsigned char, variance> <<< scaled_height, scaled_width, (scaled_width+1) * sizeof(float) >>> (orig_img_gpu, float_img_gpu, integralimages_gpu[0], scaled_width, 1);
	CHECK(cudaDeviceSynchronize());

#if TEST
	//TESTING CODE
	float * ii_cpu = (float *)malloc(sizeof(float)*(scaled_width+1)*(scaled_height+1));

	CHECK(cudaMemcpy(ii_cpu, integralimages_gpu[0], sizeof(float)*(scaled_width + 1)*(scaled_height + 1), cudaMemcpyDeviceToHost));

	FILE * ii_file = fopen("rowsum.txt", "w");
	for (int i = 0; i < scaled_height + 1; i++) {
		for (int j = 0; j < scaled_width + 1; j++) {
			fprintf(ii_file, "%1.0f", ii_cpu[i*(scaled_width + 1) + j]);
			if (j != scaled_width) {
				fprintf(ii_file, ",");
			}
		}
		fprintf(ii_file, "\n");
	}
	fclose(ii_file);
	//END TESTING CODE
#endif

	colSum << <scaled_width+1, scaled_height, (scaled_height)*sizeof(float)>> > (integralimages_gpu[0], scaled_height, scaled_width+1);
	CHECK(cudaDeviceSynchronize());
#if TEST
	CHECK(cudaMemcpy(ii_cpu, integralimages_gpu[0], sizeof(float)*(scaled_width + 1)*(scaled_height + 1), cudaMemcpyDeviceToHost));
	FILE * ii2_file = fopen("ii.txt", "w");
	for (int i = 0; i < scaled_height + 1; i++) {
		for (int j = 0; j < scaled_width + 1; j++) {
			fprintf(ii2_file, "%1.0f", ii_cpu[i*(scaled_width + 1) + j]);
			if (j != scaled_width) {
				fprintf(ii2_file, ",");
			}
		}
		fprintf(ii2_file, "\n");
	}
	fclose(ii2_file);
#endif

	for (int i = 1; i < *depth; i++) {
		scaled_width = round(scaled_width / scale);
		scaled_height = round(scaled_height / scale);

		downsampleAndRow < float , variance > << < scaled_height, scaled_width, (scaled_width + 1) * sizeof(float) >> > (float_img_gpu, float_img_gpu, integralimages_gpu[i], scaled_width, 1);
		CHECK(cudaDeviceSynchronize());

		colSum << <scaled_width + 1, scaled_height, (scaled_height) * sizeof(float) >> > (integralimages_gpu[i], scaled_height, scaled_width + 1);
		CHECK(cudaDeviceSynchronize());
	}

#if TEST
	free(ii_cpu);
#endif
	cudaFree(float_img_gpu);
		
	//Delete the original image on the GPU
	return integralimages_gpu;
}

template <class T, bool variance>
__global__ void downsampleAndRow(T * input, float * float_img, float * output, unsigned int width, float scale) {
	unsigned int idx = threadIdx.x;
	unsigned int row = blockIdx.x;

	//Determine scaled measurements
	unsigned int row_scaled = (unsigned int)(round(scale*blockIdx.x));
	unsigned int width_scaled = round(width / scale);

	//blockIdx.x should be 1 for now
	unsigned int idx_scaled = (unsigned int)(round(scale*idx));

	//SMEM is 2 x width (dynamically allocated during the kernel call)
	extern __shared__ float smem[];

	//Assign input values to shared memory, scale with nearest neighbor
	float temp = (float) input[idx_scaled + row_scaled*width];
	if (variance) {
		smem[idx] = temp*temp;
	}else{
		smem[idx] = (float)input[idx_scaled + row_scaled*width];
	}
	float_img[idx + blockIdx.x*width_scaled] = temp;
	
	//Determine the row cumulative sum for the scaled matrix
	rowSum(smem, width_scaled);

	//Assign the values to global memory
	output[idx+blockIdx.x*(width_scaled+1)] = smem[idx];
	if (threadIdx.x == 0) {
		output[width_scaled + blockIdx.x*(width_scaled + 1)] = smem[width_scaled];
	}
}

__device__ void rowSum(float * input, unsigned int width) {
	unsigned int width_offset = 0;
	unsigned int pow2 = 0;
	__shared__ float totalRemain;
	if (threadIdx.x == 0) {
		totalRemain = 0;
	}
	//unsigned int totalRemain = 0;
	while (width-width_offset > 0){
		__syncthreads();
		unsigned int workingRemain = totalRemain;
		unsigned int dist = 1;
		pow2 = smallPow2(width - width_offset);
		
		//DOWN TREE
		for (int i = pow2 / 2; i > 0; i /= 2) {
			__syncthreads();
			if (threadIdx.x < i) {
				input[width_offset + dist*(2 * threadIdx.x + 2) - 1] += input[width_offset + dist*(2 * threadIdx.x + 1) - 1];
			}
			dist *= 2;
		}

		//SET LAST VALUE AND REMAINDER
		if (threadIdx.x == 0) {
			totalRemain += input[width_offset + pow2 - 1];
			input[width_offset + pow2 - 1] = 0;
		}

		
		//UP TREE
		for (int i = 1; i < pow2; i *= 2) {
			dist /= 2;
			__syncthreads();
			if (threadIdx.x < i) {
				float swap = input[width_offset + dist*(2 * threadIdx.x + 1) - 1];
				input[width_offset + dist*(2 * threadIdx.x + 1) - 1] = input[width_offset + dist*(2 * threadIdx.x + 2) - 1];
				input[width_offset + dist*(2 * threadIdx.x + 2) - 1] += swap;
			}
		}

		__syncthreads();
		//ADD REMAINDER FROM PREVIOUS POWER OF 2	
		if (threadIdx.x < pow2) {
			input[width_offset + threadIdx.x] += workingRemain;
		}
		
		width_offset += pow2;
	}

	//SET LAST VALUE TO REMAINDER
	if (threadIdx.x == 0) {
		input[width] = totalRemain;
	}
}

__global__ void colSum(float * arr, unsigned int height, unsigned int width) {
	extern __shared__ float input[];
	input[threadIdx.x] = arr[threadIdx.x*width + blockIdx.x];
	unsigned int height_offset = 0;
	unsigned int pow2 = 0;
	__shared__ float totalRemain;
	if (threadIdx.x == 0) {
		totalRemain = 0;
	}
	//unsigned int totalRemain = 0;
	while (height - height_offset > 0) {
		__syncthreads();
		unsigned int workingRemain = totalRemain;
		unsigned int dist = 1;
		pow2 = smallPow2(height - height_offset);

		//DOWN TREE
		for (int i = pow2 / 2; i > 0; i /= 2) {
			__syncthreads();
			if (threadIdx.x < i) {
				input[height_offset + dist*(2 * threadIdx.x + 2) - 1] += input[height_offset + dist*(2 * threadIdx.x + 1) - 1];
			}
			dist *= 2;
		}

		//SET LAST VALUE AND REMAINDER
		if (threadIdx.x == 0) {
			totalRemain += input[height_offset + pow2 - 1];
			input[height_offset + pow2 - 1] = 0;
		}


		//UP TREE
		for (int i = 1; i < pow2; i *= 2) {
			dist /= 2;
			__syncthreads();
			if (threadIdx.x < i) {
				float swap = input[height_offset + dist*(2 * threadIdx.x + 1) - 1];
				input[height_offset + dist*(2 * threadIdx.x + 1) - 1] = input[height_offset + dist*(2 * threadIdx.x + 2) - 1];
				input[height_offset + dist*(2 * threadIdx.x + 2) - 1] += swap;
			}
		}

		__syncthreads();
		//ADD REMAINDER FROM PREVIOUS POWER OF 2	
		if (threadIdx.x < pow2) {
			input[height_offset + threadIdx.x] += workingRemain;
		}

		height_offset += pow2;
	}

	arr[threadIdx.x*width + blockIdx.x] = input[threadIdx.x];
	//SET LAST VALUE TO REMAINDER
	if (threadIdx.x == 0) {
		arr[height*width + blockIdx.x] = totalRemain;
	}
}

//Find the next smallest 2^N
__inline__ __device__ unsigned int smallPow2(unsigned int width) {
	width = width | (width >> 1);
	width = width | (width >> 2);
	width = width | (width >> 4);
	width = width | (width >> 8);
	width = width | (width >> 16);
	return width - (width >> 1);
}