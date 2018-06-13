#pragma once
#ifndef cuNNII
#define cuNNII
#include <stdio.h>
#include "cuda_error_check.h"
#include <cuda_runtime.h>

__inline__ __device__ unsigned int findPrev2N(unsigned int width);
template<typename T> __global__ void colSum(T * pyramid, unsigned int * sizes, unsigned int depth);
template<typename T> __device__ void rowSum(T * input, unsigned int width);
template<typename T> __global__ void downsampleRowSum(unsigned char * img, T * pyramid, unsigned int * gpu_sizes, unsigned int depth, unsigned int width, unsigned int height, float scale, float init_scale, bool variance);
template <typename T> T ** generateImagePyramid_new(unsigned char * original, unsigned int ** sizes, unsigned int * pyramidSize, unsigned int * depth, unsigned int min_size, unsigned int width, unsigned int height, float scale, float init_scale, bool variance);

//==================================================================
//                    generateImagePyramid_new						
//     This function calls integral image pyramid kernels on GPU.
//
//		Returns a host pointer to device pointers for each level
//		of the integral image pyramid.
//------------------------------------------------------------------
// T:			The type of the integral image on GPU
// original:	A device pointer to the original image
// sizes:		Empty pointer to the sizes in caller
// pyramidSize:	Pointer to the pyramid size in caller
// depth:		Pointer to the depth in caller
// min_size:	Minimum pyramid dimension allowed
// width:		The width of the original image
// height:		The height of the original image
// scale:		The pyramid scaling factor
// init_scale:	The initial level scaling factor
// variance:	Generates a squared integral image pyramid if true
//==================================================================
template <typename T> T ** generateImagePyramid_new(unsigned char * original, unsigned int ** sizes, unsigned int * pyramidSize, unsigned int * depth, unsigned int min_size, unsigned int width, unsigned int height, float scale, float init_scale, bool variance) {

	//============================================
	//  Compute the sizes of each pyramid level
	//============================================

	//Start at initialially scaled size
	float scaled_width = (1.0*width) / init_scale;
	float scaled_height = (1.0*height) / init_scale;

	//Initialize pyramid size at 0
	*pyramidSize = 0;

	//The image on GPU
	unsigned char * img_gpu;

	//Loop until width/height is less than window size
	*depth = 0;
	while (scaled_width > min_size && scaled_height > min_size) {
		//Increment depth
		(*depth)++;

		//Add to total pyramid size
		*pyramidSize += ((unsigned int)(scaled_height))*((unsigned int)(scaled_width));

		//Downscale width and height
		scaled_width /= scale;
		scaled_height /= scale;
	}

#if TEST == 1
	printf("Pyramid depth is %u\n", *depth);
	printf("Pyramid size is %u\n", sizeof(unsigned int)*(*pyramidSize));
#endif

	//Allocate sizes array on host
	*sizes = (unsigned int *)malloc(sizeof(unsigned int) * 2 * (*depth));

	//Set to initially scaled size
	scaled_width = (1.0*width) / init_scale;
	scaled_height = (1.0*height) / init_scale;

	//Loop through depth to assign sizes
	for (int i = 0; i < *depth; i++) {
		//Save sizes to sizes array
		(*sizes)[2 * i] = (unsigned int)scaled_width;
		(*sizes)[2 * i + 1] = (unsigned int)scaled_height;
		scaled_width /= scale;
		scaled_height /= scale;
	}

	//============================================
	//           Transfer sizes to GPU
	//============================================

	//Device pointer for pyramid level sizes
	unsigned int * gpu_sizes;
	//Allocate pyramid level sizes on GPU and copy sizes from host
	CHECK(cudaMalloc(&gpu_sizes, sizeof(unsigned int)*(*depth) * 2));
	CHECK(cudaMemcpy(gpu_sizes, *sizes, sizeof(unsigned int)*(*depth) * 2, cudaMemcpyHostToDevice));

#if TEST == 1
	printf("Finished saving scales\n");
#endif

	//============================================
	//       Allocate pyramid space on GPU
	//============================================

	T * imgPyramid1D_gpu;
	CHECK(cudaMalloc(&imgPyramid1D_gpu, sizeof(T)*(*pyramidSize)));
	CHECK(cudaDeviceSynchronize());
#if TEST == 1
	printf("Finished allocating pyramid space on GPU\n");
#endif

	//============================================
	//Generate host pointers to each pyramid depth
	//============================================

	//Declare pointers to each level of pyramid
	T ** imgPyramid2D = new T *[*depth];
	//Assign pointers to each pyramid depth
	imgPyramid2D[0] = imgPyramid1D_gpu;
	for (int i = 1; i < *depth; i++) {
		imgPyramid2D[i] = imgPyramid2D[i - 1] + (*sizes)[2 * (i - 1)] * (*sizes)[2 * (i - 1) + 1];
	}

#if TEST == 1
	printf("Finished setting double pointers\n");
#endif

	//============================================
	//      Copy original image to GPU
	//============================================

	//COPY image to GPU
	CHECK(cudaMalloc(&img_gpu, sizeof(unsigned char)*width*height));
	CHECK(cudaMemcpy(img_gpu, original, sizeof(unsigned char)*width*height, cudaMemcpyHostToDevice));


	//============================================
	//         Generate image pyramid 
	//         and cumulative row sum
	//============================================
	//Threads go along columns
	unsigned int minVal = min((*sizes)[0], 1024);
	dim3 block = dim3(minVal);
	//y-blocks go along rows
	//x-blocks go along multiple columns
	dim3 grid = dim3(((*sizes)[0] + 1023) / 1024, (*sizes)[1]);
#if TEST == 1
	printf("Block dimensions: (%u, %u)\n", block.x, block.y);
	printf("Grid dimensions: (%u, %u)\n", grid.x, grid.y);
#endif

	//Call image pyramid and row sum generation kernel
	downsampleRowSum<T> << <grid, block >> >(img_gpu, imgPyramid1D_gpu, gpu_sizes, *depth, width, height, scale, init_scale, variance);
	CHECK(cudaDeviceSynchronize());

	//============================================
	//       Compute cumulative column sum
	//============================================
	//Threads are along rows
	minVal = min((*sizes)[1], 1024);
	block = dim3(minVal);
	//y-blocks are along columns
	//x-blocks are along multiple rows
	grid = dim3(((*sizes)[1] + 1023) / 1024, (*sizes)[0]);
#if TEST == 1
	printf("Block dimensions: (%u, %u)\n", block.x, block.y);
	printf("Grid dimensions: (%u, %u)\n", grid.x, grid.y);
#endif

	//Call cumulative column sum kernel
	colSum<T> << <grid, block >> >(imgPyramid1D_gpu, gpu_sizes, *depth);
	CHECK(cudaDeviceSynchronize());

	cudaFree(img_gpu);

	return imgPyramid2D;
}

//==================================================================
//                    downsampleRowSum						
//		This function generates row sum pyramid on GPU.
//
//------------------------------------------------------------------
// T:			The type of the integral image on GPU
// img:			A device pointer to the original image
// pyramid:		Pointer to the image pyramid in global memory
// gpu_sizes:	Pointer to the sizes in global memory
// depth:		Pointer to the depth in caller
// width:		Width of the original image
// height:		The height of the original image
// scale:		The pyramid scaling factor
// init_scale:	The initial level scaling factor
// variance:	Generates a squared integral image pyramid if true
//==================================================================
template<typename T> __global__ void downsampleRowSum(unsigned char * img, T * pyramid, unsigned int * gpu_sizes, unsigned int depth, unsigned int width, unsigned int height, float scale, float init_scale, bool variance) {

	//Shared memory to store block row
	__shared__ T shareMem[1025];
	float curr_scale = init_scale;

	unsigned int offset = 0;

	//For each depth level, downsample and generate the row sum
	for (int i = 0; i < depth; i++) {

		//Get the current width and height from global memory
		unsigned int curr_width = (unsigned int)__ldg(&gpu_sizes[2 * i]);
		unsigned int curr_height = (unsigned int)__ldg(&gpu_sizes[2 * i + 1]);

		//If 
		if (threadIdx.x + blockIdx.x*blockDim.x < curr_width
			&& threadIdx.y + blockIdx.y*blockDim.y < curr_height) {

			unsigned int row = (unsigned int)blockIdx.y * curr_scale;
			unsigned int col = (unsigned int)threadIdx.x * curr_scale;

			T temp = (T)__ldg(&img[col + width*row]);

			if (variance) {
				shareMem[threadIdx.x] = temp*temp;
			}
			else {
				shareMem[threadIdx.x] = temp;
			}

			//GENERATE ROW SUM FOR CURRENT DEPTH
			rowSum<T>(shareMem, curr_width);

			__syncthreads();
			//COPY INTEGRAL IMAGE ROW TO GLOBAL MEMORY PYRAMID
			pyramid[offset + threadIdx.x + curr_width*blockIdx.y] = shareMem[threadIdx.x + 1];

		}
		offset += curr_width*curr_height;
		curr_scale *= scale;
	}
}

//==================================================================
//                    downsampleRowSum						
//		This function generates row sum pyramid on GPU.
//
//------------------------------------------------------------------
// T:			The type of the integral image on GPU
// img:			A device pointer to the original image
// pyramid:		Pointer to the image pyramid in global memory
// gpu_sizes:	Pointer to the sizes in global memory
// depth:		Pointer to the depth in caller
// width:		Width of the original image
// height:		The height of the original image
// scale:		The pyramid scaling factor
// init_scale:	The initial level scaling factor
// variance:	Generates a squared integral image pyramid if true
//==================================================================
template<typename T>
__device__ void rowSum(T * input, unsigned int width) {
	unsigned int width_offset = 0;
	unsigned int pow2 = 0;
	__shared__ T totalRemain;
	if (threadIdx.x == 0) {
		totalRemain = 0;
	}
	//unsigned int totalRemain = 0;
	while (width - width_offset > 0) {
		unsigned int workingRemain = totalRemain;
		unsigned int dist = 1;
		pow2 = findPrev2N(width - width_offset);

		//DOWN TREE
		for (int i = pow2 / 2; i > 0; i /= 2) {
			__syncthreads();
			if (threadIdx.x < i) {
				input[width_offset + dist*(2 * threadIdx.x + 2) - 1] += input[width_offset + dist*(2 * threadIdx.x + 1) - 1];
			}
			dist *= 2;
		}

		//SET LAST VALUE AND REMAINDER
		__syncthreads();
		if (threadIdx.x == 0) {
			totalRemain += input[width_offset + pow2 - 1];
			input[width_offset + pow2 - 1] = 0;
		}


		//UP TREE
		for (int i = 1; i < pow2; i *= 2) {
			dist /= 2;
			__syncthreads();
			if (threadIdx.x < i) {
				unsigned int swap = input[width_offset + dist*(2 * threadIdx.x + 1) - 1];
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

template<typename T>
__global__ void colSum(T * pyramid, unsigned int * sizes, unsigned int depth) {
	__shared__ T shareCol[1025];
	unsigned int offset = 0;

	for (int m = 0; m < depth; m++) {
		unsigned int curr_width = (unsigned int)__ldg(&sizes[2 * m]);
		unsigned int curr_height = (unsigned int)__ldg(&sizes[2 * m + 1]);

		if (threadIdx.x + blockIdx.x*blockDim.x < curr_height
			&& threadIdx.y + blockIdx.y*blockDim.y < curr_width) {

			shareCol[threadIdx.x] = pyramid[offset + threadIdx.x*curr_width + blockIdx.y];

			unsigned int height_offset = 0;
			unsigned int pow2 = 0;
			__shared__ T totalRemain;
			if (threadIdx.x == 0) {
				totalRemain = 0;
			}
			//unsigned int totalRemain = 0;
			while (curr_height - height_offset > 0) {
				unsigned int workingRemain = totalRemain;
				unsigned int dist = 1;
				pow2 = findPrev2N(curr_height - height_offset);

				//DOWN TREE
				for (int i = pow2 / 2; i > 0; i /= 2) {
					__syncthreads();
					if (threadIdx.x < i) {
						shareCol[height_offset + dist*(2 * threadIdx.x + 2) - 1] += shareCol[height_offset + dist*(2 * threadIdx.x + 1) - 1];
					}
					dist *= 2;
				}

				__syncthreads();
				//SET LAST VALUE AND REMAINDER
				if (threadIdx.x == 0) {
					totalRemain += shareCol[height_offset + pow2 - 1];
					shareCol[height_offset + pow2 - 1] = 0;
				}


				//UP TREE
				for (int i = 1; i < pow2; i *= 2) {
					dist /= 2;
					__syncthreads();
					if (threadIdx.x < i) {
						unsigned int swap = shareCol[height_offset + dist*(2 * threadIdx.x + 1) - 1];
						shareCol[height_offset + dist*(2 * threadIdx.x + 1) - 1] = shareCol[height_offset + dist*(2 * threadIdx.x + 2) - 1];
						shareCol[height_offset + dist*(2 * threadIdx.x + 2) - 1] += swap;
					}
				}

				__syncthreads();
				//ADD REMAINDER FROM PREVIOUS POWER OF 2	
				if (threadIdx.x < pow2) {
					shareCol[height_offset + threadIdx.x] += workingRemain;
				}

				height_offset += pow2;
			}

			//SET LAST VALUE TO REMAINDER
			if (threadIdx.x == 0) {
				shareCol[curr_height] = totalRemain;
			}

			__syncthreads();
			pyramid[offset + threadIdx.x*curr_width + blockIdx.y] = shareCol[threadIdx.x + 1];
			offset += curr_width*curr_height;
		}
	}
	free(sizes);
}

//Find the next smallest 2^N
__device__ unsigned int findPrev2N(unsigned int width) {
	width = width | (width >> 1);
	width = width | (width >> 2);
	width = width | (width >> 4);
	width = width | (width >> 8);
	width = width | (width >> 16);
	return width - (width >> 1);
}

#endif