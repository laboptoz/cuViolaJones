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
	float * ii_cpu = (float *)malloc(sizeof(float)*(scaled_width+1)*(scaled_height+1));
	//CHECK(cudaMemcpy(ii_cpu, float_img_gpu, sizeof(float)*scaled_width*scaled_height, cudaMemcpyDeviceToHost));
	/*for (int i = 0; i < scaled_height; i++) {
		for (int j = 0; j < scaled_width; j++) {
			printf("%1.0f\t", ii_cpu[i*scaled_width + j]);
		}
		printf("\n");
	}
	printf("\n");*/
	CHECK(cudaMemcpy(ii_cpu, integralimages_gpu[0], sizeof(float)*(scaled_width + 1)*(scaled_height + 1), cudaMemcpyDeviceToHost));
	/*(for (int i = 0; i < scaled_height+1; i++) {
		for (int j = 0; j < scaled_width+1; j++) {
			printf("%1.0f\t", ii_cpu[i*(scaled_width+1) + j]);
		}
		printf("\n");
	}*/

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


	printf("Finished testing\n");

#endif

	colSum << <scaled_width+1, scaled_height, (scaled_height)*sizeof(float)>> > (integralimages_gpu[0], scaled_height, scaled_width+1);
	CHECK(cudaMemcpy(ii_cpu, integralimages_gpu[0], sizeof(float)*(scaled_width + 1)*(scaled_height + 1), cudaMemcpyDeviceToHost));
	/*(for (int i = 0; i < scaled_height+1; i++) {
	for (int j = 0; j < scaled_width+1; j++) {
	printf("%1.0f\t", ii_cpu[i*(scaled_width+1) + j]);
	}
	printf("\n");
	}*/

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


	printf("Finished testing\n");
	printf("Finished col sum");
	//ANYTHING BELOW THIS PROBABLY DOESNT WORK
	/*dim3 unit_colblock(scaled_height);
	dim3 unit_colgrid(1, scaled_width);  //DOES NOT SUPPORT LARGE IMAGES RIGHT NOW
	colSum <<<unit_colgrid, unit_colblock, scaled_height * sizeof(float) >> > (integralimages_gpu[0], integralimages_gpu[0], scaled_height, scaled_width);
	CHECK(cudaDeviceSynchronize());
	cudaFree(orig_img_gpu);

	//GENERATE REMAINING INTEGRAL IMAGES

	for (int i = 1; i < *depth; i++) {
		scaled_width = round(scaled_width / scale);
		scaled_height = round(scaled_height / scale);
		dim3 rowblock = dim3(scaled_width);
		dim3 rowgrid = dim3(1, scaled_height);

		downsampleAndRow <float, variance> <<< rowgrid, rowblock, scaled_width * sizeof(float) >> > (float_img_gpu, float_img_gpu, integralimages_gpu[i], scaled_width, 1);
		CHECK(cudaDeviceSynchronize());

		dim3 colblock = dim3(scaled_height);
		dim3 colgrid = dim3(1, scaled_width);
		colSum << <colgrid, colblock, scaled_height * sizeof(float) >> > (integralimages_gpu[i], integralimages_gpu[i], scaled_height, scaled_width);
		CHECK(cudaDeviceSynchronize());

	}*/
	cudaFree(float_img_gpu);

#if TEST
	free(ii_cpu);
#endif
		
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












































/*
__global__ void colSum(float * in_arr, float * out_arr, unsigned int height, unsigned int width) {
	//The shared memory
	extern __shared__ float input[];
	unsigned int idx = threadIdx.x;

	//Copy data into shared memory
	input[idx] = (float)in_arr[idx*width + blockIdx.y];
	input[height + idx] = input[idx];

	unsigned int height_offset = 0;
	unsigned int remainder = 0;

	//Loop through 2^N sized segments of the columns
	while (height_offset < height) {
		unsigned int offset = 1;

		//Find next smallest 2^N
		unsigned int pw2 = smallPow2(height - height_offset);
		unsigned int working_rem = remainder;

		//If pw2 is 1, don't go through this whole process
		if (pw2 != 1) {

			//Loop through the binary tree
			for (int i = pw2 / 2; i > 0; i /= 2) {
				__syncthreads();
				if (idx < i) {
					input[height + offset * (2 * idx + 2) - 1 + height_offset] += input[height + offset * (2 * idx + 1) - 1 + height_offset];
				}
				offset *= 2;
			}

			//Add the highest value as the remainder
			remainder += input[height + pw2 + height_offset - 1];

			//Set highest value to zero
			if (idx == 0) {
				input[height + pw2 + height_offset - 1] = 0;
			}

			//Reverse through the binary tree
			for (int i = 1; i < pw2; i *= 2) {
				offset /= 2;
				__syncthreads();
				if (idx < i) {

					//Swap and add
					float swap = input[height + offset*(2 * idx + 1) - 1 + height_offset];
					input[height + offset*(2 * idx + 1) - 1 + height_offset] = input[height + offset*(2 * idx + 2) - 1 + height_offset];
					input[height + offset*(2 * idx + 2) - 1 + height_offset] += swap;
				}
			}

			//Add original values and remainder
			if (idx < pw2) {
				input[height + idx + height_offset] += input[idx + height_offset] + working_rem;
			}
		}
		else {
			//Set to original values and remainder
			if (idx < pw2) {
				input[height + idx + height_offset] = input[idx + height_offset] + working_rem;
			}
		}
		__syncthreads();
		height_offset += pw2;
	}

	//Move shared memory to input array
	out_arr[idx*width + blockIdx.y] = input[height + idx];
}*/

//Find the next smallest 2^N
__inline__ __device__ unsigned int smallPow2(unsigned int width) {
	width = width | (width >> 1);
	width = width | (width >> 2);
	width = width | (width >> 4);
	width = width | (width >> 8);
	width = width | (width >> 16);
	return width - (width >> 1);
}