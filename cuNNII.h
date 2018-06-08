#include <cuda_runtime.h>
#include "cuda_error_check.h"
#define TEST 0

__device__ void rowSum(float * input, unsigned int width);
__inline__ __device__ unsigned int smallPow2(unsigned int width);
template <class T> __global__ void downsampleAndRow(T * input, float * float_img, float * output, unsigned int width, float scale);
__global__ void colSum(float * in_arr, float * out_arr, unsigned int height, unsigned int width);

template<bool variance>
float ** generateImagePyramid(unsigned char * original, unsigned int ** sizes_ptr, unsigned int * depth, unsigned int min_size, unsigned int width, unsigned int height, float scale) {
	printf("Generating image pyramid on GPU\n");
	//CUDA MALLOC AND COPY THE ORIGINAL IMAGE
	unsigned char * orig_img_gpu;
	CHECK(cudaMalloc(&orig_img_gpu, sizeof(unsigned char)*width*height));
	CHECK(cudaMemcpy(orig_img_gpu, original, sizeof(unsigned char)*width*height, cudaMemcpyHostToDevice));

	float * float_img_gpu;
	CHECK(cudaMalloc(&float_img_gpu, sizeof(unsigned char)*width*height));

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
		sum_size += scaled_width*scaled_height;
		scaled_width = round(scaled_width / scale);
		scaled_height = round(scaled_height / scale);
	}
	printf("Image pyramid depth is: %u\n", *depth);
	
	//CUDA MALLOC THE IMAGES
	float * integralImagePyramid_gpu;
	float * imagePyramid_gpu;
	CHECK(cudaMalloc(&imagePyramid_gpu, sum_size * sizeof(float)));
	CHECK(cudaMalloc(&integralImagePyramid_gpu, sum_size * sizeof(float)));

	float ** integralimages_gpu = new float *[*depth];

	//Assign pyramid sizes to argument 'sizes'
	sum_size = 0;
	scaled_width = width;
	scaled_height = height;
	*sizes_ptr = (unsigned int *) malloc(2*(*depth)*sizeof(unsigned int));
	unsigned int * sizes = *sizes_ptr;
	for (int i = 0; i < *depth; i++) {
		integralimages_gpu[i] = imagePyramid_gpu + sum_size;
		sum_size += scaled_width*scaled_height;
		sizes[2*i] = scaled_width;
		sizes[2*i + 1] = scaled_height;
		scaled_width = round(scaled_width / scale);
		scaled_height = round(scaled_height / scale);
	}
	

		//CHECK(cudaMalloc(&img_gpu, sizeof(unsigned char)*height*width));
		//CHECK(cudaMalloc(&ii_gpu, sizeof(float)  * scaled_height * scaled_width));

	//GENERATE SIZE 1 INTEGRAL IMAGE
	scaled_width = width;
	scaled_height = height;
	dim3 unit_rowblock(scaled_width);
	dim3 unit_rowgrid(1, scaled_height);  //DOES NOT SUPPORT LARGE IMAGES RIGHT NOW
	
	downsampleAndRow <unsigned char, variance> <<< unit_rowgrid, unit_rowblock, 2 * scaled_width * sizeof(float) >>> (orig_img_gpu, float_img_gpu, integralimages_gpu[0], scaled_width, 1);
	CHECK(cudaDeviceSynchronize());

	dim3 unit_colblock(scaled_height);
	dim3 unit_colgrid(1, scaled_width);  //DOES NOT SUPPORT LARGE IMAGES RIGHT NOW
	colSum <<<unit_colgrid, unit_colblock, 2 * scaled_height * sizeof(float) >> > (integralimages_gpu[0], integralimages_gpu[0], scaled_height, scaled_width);
	CHECK(cudaDeviceSynchronize());
	cudaFree(orig_img_gpu);


	//GENERATE REMAINING INTEGRAL IMAGES
#if TEST
		float * ii_cpu = new float[sizes[0] * sizes[1]];
		printf("\nPyramid level: 0\n");
		CHECK(cudaMemcpy(ii_cpu, integralimages_gpu[0], sizeof(float) * scaled_width * scaled_height, cudaMemcpyDeviceToHost));
		for (int ik = 0; ik < scaled_height; ik++) {
			for (int j = 0; j < scaled_width; j++) {
				printf("%f\t", ii_cpu[ik*scaled_width + j]);
			}
			printf("\n");
		}
		printf("\n");
#endif

	for (int i = 1; i < *depth; i++) {
#if TEST
		printf("Pyramid level: %u\n", i);
#endif
		scaled_width = round(scaled_width / scale);
		scaled_height = round(scaled_height / scale);
		dim3 rowblock = dim3(scaled_width);
		dim3 rowgrid = dim3(1, scaled_height);

		downsampleAndRow <float, variance> <<< rowgrid, rowblock, 2 * scaled_width * sizeof(float) >> > (float_img_gpu, float_img_gpu, integralimages_gpu[i], scaled_width, 1);
		CHECK(cudaDeviceSynchronize());

		dim3 colblock = dim3(scaled_height);
		dim3 colgrid = dim3(1, scaled_width);
		colSum << <colgrid, colblock, 2 * scaled_height * sizeof(float) >> > (integralimages_gpu[i], integralimages_gpu[i], scaled_height, scaled_width);
		CHECK(cudaDeviceSynchronize());

#if TEST
		CHECK(cudaMemcpy(ii_cpu, integralimages_gpu[i], sizeof(float) * scaled_width * scaled_height, cudaMemcpyDeviceToHost));
		for (int ik = 0; ik < scaled_height; ik++) {
			for (int j = 0; j < scaled_width; j++) {
				printf("%f\t", ii_cpu[ik*scaled_width + j]);
			}
			printf("\n");
		}
		printf("\n");
#endif
	}
	cudaFree(float_img_gpu);
	


	

#if TEST
	printf("Finished testing\n");
#endif

	
	//Delete the original image on the GPU
	return integralimages_gpu;
}

template <class T, bool variance>
__global__ void downsampleAndRow(T * input, float * float_img, float * output, unsigned int width, float scale) {
	unsigned int idx = threadIdx.x;
	unsigned int row = blockIdx.y;

	//Determine scaled measurements
	unsigned int row_scaled = (unsigned int)(round(scale*blockIdx.y));
	unsigned int width_scaled = round(width / scale);

	//blockIdx.x should be 1 for now
	unsigned int idx_scaled = (unsigned int)(round(scale*idx) + blockIdx.x*1024);

	//SMEM is 2 x width (dynamically allocated during the kernel call)
	extern __shared__ float smem[];

	//Assign input values to shared memory, scale with nearest neighbor
	float temp = (float) input[idx_scaled + row_scaled*width];
	if (variance) {
		smem[idx] = temp*temp;
	}else{
		smem[idx] = temp;
	}
	float_img[idx + blockIdx.y*width_scaled] = temp;

	//Copy smem to second half of shared memory
	smem[width_scaled + idx] = smem[idx];

	//Determine the row cumulative sum for the scaled matrix
	rowSum(smem, width_scaled);

	//Assign the values to global memory
	output[idx+blockIdx.y*width_scaled] = smem[width_scaled + idx];
}

__device__ void rowSum(float * input, unsigned int width) {
	unsigned int idx = threadIdx.x;

	//How much of the width is processed
	unsigned int width_offset = 0;

	//The last value of the previous width segment
	unsigned int remainder = 0;

	//Loops over largest possible segments of 2^N
	while (width_offset < width) {

		//Offset is the offset between added samples
		unsigned int offset = 1;

		//The largest available power of 2
		unsigned int pw2 = smallPow2(width-width_offset);

		//The previous remainder value
		unsigned int working_rem = remainder;

		//If pw2 is not 1
		if (pw2 != 1) {

			//Loop through binary addition tree
			for (int i = pw2 / 2; i > 0; i /= 2) {
				__syncthreads();
				if (idx < i) {
					input[width + offset * (2 * idx + 2) - 1 + width_offset] += input[width + offset * (2 * idx + 1) - 1 + width_offset];
				}

				//Increase the offset between sample additions
				offset *= 2;
			}

			//Set the remainder to be the last value
			remainder += input[width + pw2 + width_offset - 1];

			//Set last value in segment to zero
			if (idx == 0) {
				input[width + pw2 + width_offset - 1] = 0;
			}

			//Reverse the binary tree
			for (int i = 1; i < pw2; i *= 2) {
				offset /= 2;
				__syncthreads();
				if (idx < i) {

					//Swap and add
					float swap = input[width + offset*(2 * idx + 1) - 1 + width_offset];
					input[width + offset*(2 * idx + 1) - 1 + width_offset] = input[width + offset*(2 * idx + 2) - 1 + width_offset];
					input[width + offset*(2 * idx + 2) - 1 + width_offset] += swap;

				}
			}

			//Add the original values and the working remainder
			if (idx < pw2) {
				input[width + idx + width_offset] += input[idx + width_offset] + working_rem;
			}
		} else {
			//Set to the original values and the working remainder
			if (idx < pw2) {
				input[width + idx + width_offset] = input[idx + width_offset] + working_rem;
			}
		}
		__syncthreads();

		//Adjust the width offset
		width_offset += pw2;
	}
	
}

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