#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include "paths.hpp"
#include "haar.h"
#include "cuNNII_v2.cuh"
#include "Filter.cuh"
//#include "cuda_error_check.h"
#include "parameter_loader.h"

#include <iostream>
#include <fstream>

// compute integral images
void integralImages_cpp(ImageUnion *src, ImageUnion *sum, ImageUnion *sqsum);

// compute down-scaled image
void nearestNeighbor_cpp(ImageUnion *src, ImageUnion *dst);

// rounding function
inline  int  myRound_cpp(float value){
	return (int)(value + (value >= 0 ? 0.5 : -0.5));
}

/*******************************************************
* Compute square root of integer value
*******************************************************/
__device__ unsigned int int_sqrt_cpp(unsigned int value) {
	int i;
	unsigned int a = 0, b = 0, c = 0;
	for (i = 0; i < (32 >> 1); i++)
	{
		c <<= 2;
#define UPPERBITS(value) (value>>30)
		c += UPPERBITS(value);
#undef UPPERBITS
		value <<= 2;
		a <<= 1;
		b = (a << 1) | 1;
		if (c >= b)
		{
			c -= b;
			a++;
		}
	}
	return a;
}

/*******************************************************
* Perform cascade classifier
*******************************************************/
__device__ bool operation(unsigned int* sum, unsigned int* sqsum, unsigned int idx, 
	unsigned int img_width, unsigned int wd_width, unsigned int wd_height,
	Stage * stages_gpu) {

	// normalize image by 1/std
	// index 0 top left, 1 top right, 2 bottom left, 3 bottom right
	unsigned int idx0 = idx;
	unsigned int idx1 = idx + wd_width - 1;
	unsigned int idx2 = idx + (wd_height - 1) * img_width;
	unsigned int idx3 = idx2 + wd_width - 1;
	unsigned int wd_sqsum = sqsum[idx0] - sqsum[idx1] - sqsum[idx2] + sqsum[idx3];
	unsigned int wd_mean = sum[idx0] - sum[idx1] - sum[idx2] + sum[idx3];
	int norm_factor = wd_sqsum * (wd_width * wd_height) - wd_mean * wd_mean;
	if (norm_factor > 0)
		norm_factor = int_sqrt_cpp(norm_factor);
	else
		norm_factor = 1;

	// run stages
	const int num_stages = 25;
	int num_filters;
	int stage_sum;
	bool pass = false;
	for (int i = 0; i < num_stages; i++) {
		pass = stages_gpu[i].getValue<unsigned int>(sum + idx0, norm_factor, img_width);

		// warp level pruning
		unsigned int mask = __ballot(pass == false);
		unsigned int falseCount = __popc(mask);
		if (falseCount > PRUNING)
			pass = false;
		if (!pass)
			return false;
	}
	return true;
}


/******************************************************
* Run sliding window step
******************************************************/
__global__ void slide_window(
	unsigned int* sum1, unsigned int* sqsum1,
	bool *d_activation,
	unsigned int wd_width, unsigned int wd_height, 
	unsigned int img_width, unsigned int img_height,
	Stage * stages_gpu) {

	// get top left element's index
	unsigned int topLeft_idx = blockIdx.y * img_width + blockIdx.x * blockDim.x + threadIdx.x;
	
	// to differentiate if goes outside of image
	if ((topLeft_idx + wd_width - 1) / img_width == blockIdx.y) {

		// run cascade classifier
		bool result = operation(sum1, sqsum1, topLeft_idx, img_width, wd_width, wd_height, stages_gpu);
		
		// return value to activation map
		unsigned int dest_idx = blockIdx.y * (img_width - wd_width + 1) + blockIdx.x * blockDim.x + threadIdx.x;
		d_activation[dest_idx] = result;
	}
}


/******************************************************
* Get integral images and run sliding window to detect faces
******************************************************/
void detect_faces(unsigned int img_width, unsigned int img_height, std::vector<Rectangle> &allCandidates, 
	ImageUnion* _img, float scale_factor, int minNeighbors, unsigned int * num_stages, Stage * stages_gpu) {

	// group overlaping windows
	const float GROUP_EPS = 0.4f;

	// pointer to input image
	ImageUnion *img = _img;

	// pointers for the created structs
	ImageUnion *img1 = new ImageUnion();
	ImageUnion *sum1 = new ImageUnion();
	ImageUnion *sqsum1 = new ImageUnion();

	// malloc for img1: unsigned char
	img1->width = img->width;
	img1->height = img->height;
	img1->dataChar = (unsigned char *)malloc(sizeof(unsigned char)*(img->height*img->width));

	// malloc for sum1: unsigned char
	sum1->width = img->width;
	sum1->height = img->height;
	sum1->dataInt = (int*)malloc(sizeof(int)*(img->height*img->width));

	// malloc for sqsum1: unsigned char
	sqsum1->width = img->width;
	sqsum1->height = img->height;
	sqsum1->dataInt = (int*)malloc(sizeof(int)*(img->height*img->width));

	//------------------------------------
	//------    CREATE II ON GPU  --------
	//------------------------------------

	using pyr_type = unsigned int;
#if GPUII == 1

	unsigned int * sizes = nullptr;
	unsigned int * depth = new unsigned int;
	unsigned int * pyramidSize = new unsigned int;

	pyr_type ** impyr = generateImagePyramid_new<pyr_type>(img->dataChar, &sizes, pyramidSize, depth, WIN_SIZE, img1->width, img1->height, SCALING, SCALING, false);

	unsigned int * v_sizes = nullptr;
	unsigned int * v_depth = new unsigned int;
	unsigned int * v_pyramidSize = new unsigned int;

	pyr_type ** v_impyr = generateImagePyramid_new<pyr_type>(img->dataChar, &sizes, pyramidSize, depth, WIN_SIZE, img1->width, img1->height, SCALING, SCALING, true);

#endif

	int wd_height = WIN_SIZE, wd_width = WIN_SIZE;

	int counter = 0;

#if GPUII == 1
	for (int i = 0; i < *depth; i++) {
#elif GPUII == 0
	for (int i = 0;;i++){
#endif
		// size of the image scaled up
		ImageDim winSize = { myRound_cpp(wd_width*scale_factor), myRound_cpp(wd_height*scale_factor) };
		// size of the image scaled down (from bigger to smaller)
		ImageDim sz = { (img->width / scale_factor), (img->height / scale_factor) };
		// difference between sizes of the scaled image and the original detection window
		ImageDim sz1 = { sz.width - wd_width, sz.height - wd_height };

		// if the actual scaled image is smaller than the original detection window, break
		if (sz1.width < 0 || sz1.height < 0)
			break;

		// set image dim
		img1->width = sz.width;
		img1->height = sz.height;

		// integral image dims
		sum1->width = sz.width;
		sum1->height = sz.height;
		sqsum1->width = sz.width;
		sqsum1->height = sz.height;
		nearestNeighbor_cpp(img, img1);
		integralImages_cpp(img1, sum1, sqsum1);


#if TEST == 1
		char cpu_fname[] = "__cpu_img.csv";
		cpu_fname[0] = 'a' + i;
		FILE * cpu_img = fopen(cpu_fname, "w");
		for (int j = 0; j < sz.height; j++) {
			for (int k = 0; k < sz.width; k++) {
				fprintf(cpu_img, "%u", img1->dataChar[j*sz.width + k]);
				if (k != sz.width - 1) {
					fprintf(cpu_img, ",");
				}
			}
			if (j != sz.height - 1)
				fprintf(cpu_img, "\n");
		}
		fclose(cpu_img);

		char gpu_fname[] = "__gpu_img.csv";
		gpu_fname[0] = 'a' + i;
		FILE * gpu_img = fopen(gpu_fname, "w");

		unsigned int curr_width = sizes[2 * i];
		unsigned int curr_height = sizes[2 * i + 1];
		pyr_type * curr_gpu_img = (pyr_type *)malloc(sizeof(pyr_type)*curr_width*curr_height);
		CHECK(cudaMemcpy(curr_gpu_img, impyr[i], sizeof(pyr_type)*curr_width*curr_height, cudaMemcpyDeviceToHost));
		printf("Printing level %u, size (%u, %u)\n", i, curr_width, curr_height);
		for (int j = 0; j < curr_height; j++) {
			for (int k = 0; k < curr_width; k++) {
				fprintf(gpu_img, "%u", curr_gpu_img[j*curr_width + k]);
				if (k != curr_width - 1) {
					fprintf(gpu_img, ",");
				}
			}
			if (j != curr_height - 1)
				fprintf(gpu_img, "\n");
		}
		free(curr_gpu_img);
		fclose(gpu_img);
#endif

#if GPUII == 1
		unsigned int * d_sum1 = impyr[i];
		unsigned int * d_sqsum1 = v_impyr[i];
		unsigned int down_h = sizes[2 * i + 1];//sz.height;
		unsigned int down_w = sizes[2 * i];// sz.width;
#elif GPUII == 0
		// copy cpu integral image to gpu
		unsigned int* d_sum1;
		unsigned int* d_sqsum1;
		CHECK(cudaMalloc((void**)&d_sum1, sum1->width * sum1->height * sizeof(int)));
		CHECK(cudaMalloc((void**)&d_sqsum1, sqsum1->width * sqsum1->height * sizeof(int)));
		CHECK(cudaMemcpy(d_sum1, sum1->dataInt, sum1->width * sum1->height * sizeof(int), cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_sqsum1, sqsum1->dataInt, sqsum1->width * sqsum1->height * sizeof(int), cudaMemcpyHostToDevice));

		unsigned int down_h = sz.height;
		unsigned int down_w = sz.width;
#endif

		unsigned int num_wd_col = down_h - wd_height + 1;  // number of windows on a column
		unsigned int num_wd_row = down_w - wd_width + 1;  // number of windows on a row

		// create activation map
		unsigned int act_size = num_wd_col * num_wd_row * sizeof(bool);
		bool *h_activation = (bool *)malloc(act_size);
		bool *d_activation;
		CHECK(cudaMalloc((void**)&d_activation, act_size));

		// run sliding window kernel
		dim3 blockDim(1024);
		dim3 gridDim((num_wd_row + blockDim.x - 1) / blockDim.x, down_h - wd_height + 1);
		slide_window << <gridDim, blockDim >> > (d_sum1, d_sqsum1, d_activation, wd_width, wd_height, down_w, down_h, stages_gpu);
		CHECK(cudaDeviceSynchronize());

		// copy activation map back
		CHECK(cudaMemcpy(h_activation, d_activation, act_size, cudaMemcpyDeviceToHost));

		// add activations to vector
		for (unsigned int j = 0; j < num_wd_col * num_wd_row; j++) {
			if (h_activation[j]) {
				unsigned int y = j / num_wd_row;
				unsigned int x = j % num_wd_row;
				Rectangle r = { myRound_cpp(x * scale_factor), myRound_cpp(y * scale_factor), wd_width * scale_factor, wd_height * scale_factor };
				allCandidates.push_back(r);
				counter++;
			}
		}
		// increment scale factor
		scale_factor *= 1.2;

		// free cuda var
#if GPUII == 0
		CHECK(cudaFree((void*)d_sum1));
		CHECK(cudaFree((void*)d_sqsum1));
#endif
		CHECK(cudaFree((void*)d_activation));
		free(h_activation);
	}

	// sort, clean and organize the labeled windows
	if (minNeighbors != 0) {
		groupRectangles(allCandidates, minNeighbors, GROUP_EPS);
	}

	// free image/integral image/squared integral image
	free(img1->dataChar);
	free(sum1->dataInt);
	free(sqsum1->dataInt);

	// free GPUII intrinsic elements
#if GPUII == 1
	CHECK(cudaFree(impyr[0]));
	CHECK(cudaFree(v_impyr[0]));
#endif
}


/****************************************************
* Compute the integral image on CPU
****************************************************/
void integralImages_cpp(ImageUnion *src, ImageUnion *sum, ImageUnion *sqsum) {
	int x, y, s, sq, t, tq;
	unsigned char it;
	int height = src->height;
	int width = src->width;
	unsigned char *data = src->dataChar;
	int * sumData = sum->dataInt;
	int * sqsumData = sqsum->dataInt;
	for (y = 0; y < height; y++) {
		s = 0;
		sq = 0;
		// loop over the number of columns
		for (x = 0; x < width; x++) {
			it = data[y*width + x];
			// sum of the current row (integer)
			s += it;
			sq += it*it;
			// sum rows and cols above
			t = s;
			tq = sq;
			if (y != 0) {
				t += sumData[(y - 1)*width + x];
				tq += sqsumData[(y - 1)*width + x];
			}
			sumData[y*width + x] = t;
			sqsumData[y*width + x] = tq;
		}
	}
}

/***********************************************************
* Downscale an image using nearest neighbor
**********************************************************/
void nearestNeighbor_cpp(ImageUnion *src, ImageUnion *dst) {
	int y;
	int j;
	int x;
	int i;
	unsigned char* t;
	unsigned char* p;
	int w1 = src->width;
	int h1 = src->height;
	int w2 = dst->width;
	int h2 = dst->height;
	int rat = 0;

	unsigned char* src_data = src->dataChar;
	unsigned char* dst_data = dst->dataChar;

	int x_ratio = (int)((w1 << 16) / w2) + 1;
	int y_ratio = (int)((h1 << 16) / h2) + 1;

	for (i = 0; i<h2; i++) {
		t = dst_data + i*w2;
		y = ((i*y_ratio) >> 16);
		p = src_data + y*w1;
		rat = 0;
		for (j = 0; j<w2; j++) {
			x = (rat >> 16);
			*t++ = p[x];
			rat += x_ratio;
		}
	}
}

