#pragma once
#include <cuda_runtime.h>
//#include "cuIntegralImage.cuh"
//#include "cuNNII.h"

#include "haar.h"
#include "image.h"
#include <stdio.h>
//#include "stdio-wrapper.h"
#include "paths.hpp"

#include <cuda.h>

#include "cuNNII.h"
#include "Filter.cuh"
//#include "cuda_error_check.h"
#include "parameter_loader.h"

#include <iostream>
#include <fstream>

/* compute integral images */
void integralImages_cpp(MyImage *src, MyIntImage *sum, MyIntImage *sqsum);

/* compute scaled image */
void nearestNeighbor_cpp(MyImage *src, MyImage *dst);

/* rounding function */
inline  int  myRound_cpp(float value)
{
	return (int)(value + (value >= 0 ? 0.5 : -0.5));
}

/***********************************************
* Note:
* The int_sqrt is softwar integer squre root.
* GPU has hardware for floating squre root (sqrtf).
* In GPU, it is wise to convert an int variable
* into floating point, and use HW sqrtf function.
* More info:
* http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#standard-functions
**********************************************/
/*****************************************************
* The int_sqrt is only used in runCascadeClassifier
* If you want to replace int_sqrt with HW sqrtf in GPU,
* simple look into the runCascadeClassifier function.
*****************************************************/
__device__ unsigned int int_sqrt_cpp(unsigned int value)
{
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


__device__ bool operation(unsigned int* sum, unsigned int* sqsum, unsigned int idx, 
	unsigned int img_width, unsigned int wd_width, unsigned int wd_height,
	Stage * stages_gpu) {
	// idx: top left corner index of the window

	/*
	* todo: 
	* assume we have an array of number of filters per stage - filter_counts
	* filter class - filter
	* array of pointers to filters - filters
	* sum integral image [0] pointer - sum_int
	* stage threshold - 
	*
	* update filter function to take argument of variance normalization factor
	*/ 

	// normalize image; 0 top left, 1 top right, 2 bottom left, 3 bottom right
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
	//printf("i0 %d, i1 %d, i2 %d, i3 %d, sqsum %d, mean %d, norm_factor %d, area %d\n", idx0, idx1, idx2, idx3, wd_sqsum, wd_mean, norm_factor, wd_width * wd_height);

	// run filter
	const int num_stages = 25;

	int num_filters;
	int stage_sum;

	bool pass = false;
	for (int i = 0; i < num_stages; i++) {

		//printf("stage %d: ", i);

		//printf("sums %d %d %d %d, norm factor %f, running filter, idx %d \n", *(sum + idx0), sum[idx1], sum[idx2], sum[idx3], norm_factor, idx);
		pass = stages_gpu[i].getValue<unsigned int>(sum + idx0, norm_factor, img_width);

		//break;

		if (!pass)
			return false;
	}
	//printf("result.. ");
	return true;
}

//__device__ int test_sum(int* d_image, int topLeft_idx, int img_width, float norm_factor) {
//	float val0 = d_image[topLeft_idx];
//	float val1 = d_image[topLeft_idx + 1];
//	return (int)(val0 + val1);
//}

__global__ void slide_window(
	unsigned int* sum1, unsigned int* sqsum1,
	bool *d_activation,
	unsigned int wd_width, unsigned int wd_height, 
	unsigned int img_width, unsigned int img_height,
	Stage * stages_gpu

) {
	//unsigned int param_idx = 0;
	//for (int i = 0; i < 25; i++) {
	//	unsigned int num_filters = stages_gpu[i].num_filters;
	//	for (int j = 0; j < num_filters; j++) {
	//		Filter f = stages_gpu[i].filters[j];

	//		params[param_idx] = f.x1;
	//		params[param_idx + 1] = f.y1;
	//		params[param_idx + 2] = f.width1;
	//		params[param_idx + 3] = f.height1;
	//		params[param_idx + 4] = f.weight1;

	//		params[param_idx + 5] = f.x2;
	//		params[param_idx + 6] = f.y2;
	//		params[param_idx + 7] = f.width2;
	//		params[param_idx + 8] = f.height2;
	//		params[param_idx + 9] = f.weight2;

	//		params[param_idx + 10] = f.x3;
	//		params[param_idx + 11] = f.y3;
	//		params[param_idx + 12] = f.width3;
	//		params[param_idx + 13] = f.height3;
	//		params[param_idx + 14] = f.weight3;


	//		params[param_idx + 15] = f.threshold;
	//		params[param_idx + 16] = f.alpha1;
	//		params[param_idx + 17] = f.alpha2;

	//		param_idx += 18;

	//	}
	//	params[param_idx] = stages_gpu[i].threshold;

	//	param_idx += 1;
	//}


	//printf("in slide window, ");
	// get four corners
	unsigned int topLeft_idx = blockIdx.y * img_width + blockIdx.x * blockDim.x + threadIdx.x;
	//printf("r %d c %d, ", blockIdx.y, blockIdx.x * blockDim.x + threadIdx.x);
	
	//int result = test_sum(d_image, topLeft_idx, img_width, 1);
	
	// to differentiate if goes outside of image
	if ((topLeft_idx + wd_width - 1) / img_width == blockIdx.y) {
		bool result = operation(sum1, sqsum1, topLeft_idx, img_width, wd_width, wd_height, stages_gpu);
		//printf("after operation, ");

		unsigned int dest_idx = blockIdx.y * (img_width - wd_width + 1) + blockIdx.x * blockDim.x + threadIdx.x;
		//printf("%d ,", dest_idx);
		d_activation[dest_idx] = result;
	}
	//printf("\n");
}

void detect_faces(unsigned int img_width, unsigned int img_height, std::vector<MyRect> &allCandidates, 
	MyImage* _img, float scale_factor, int minNeighbors) {
	/* group overlaping windows */
	const float GROUP_EPS = 0.4f;

	/* pointer to input image */
	MyImage *img = _img;
	/***********************************
	* create structs for images
	* see haar.h for details
	* img1: normal image (unsigned char)
	* sum1: integral image (int)
	* sqsum1: square integral image (int)
	**********************************/
	MyImage image1Obj;
	MyIntImage sum1Obj;
	MyIntImage sqsum1Obj;
	/* pointers for the created structs */
	MyImage *img1 = &image1Obj;
	MyIntImage *sum1 = &sum1Obj;
	MyIntImage *sqsum1 = &sqsum1Obj;

	/* malloc for img1: unsigned char */
	createImage(img->width, img->height, img1);
	/* malloc for sum1: unsigned char */
	createSumImage(img->width, img->height, sum1);
	/* malloc for sqsum1: unsigned char */
	createSumImage(img->width, img->height, sqsum1);

	//------------------------------------
	//------    CREATE II ON GPU  --------
	//------------------------------------
	
#if GPUII == 1
	unsigned int * ii_sizes = nullptr;
	unsigned int ii_depth = 0;
	unsigned int ** ii_gpu = generateImagePyramid<false>(img1->data, &ii_sizes, &ii_depth, 24, img1->width, img1->height, 1.2);

	unsigned int * sqii_sizes = nullptr;
	unsigned int sqii_depth = 0;
	unsigned int ** sqii_gpu = generateImagePyramid<true>(img1->data, &ii_sizes, &ii_depth, 24, img1->width, img1->height, 1.2);
#endif

	/*
	* TODO:
	* assume we have array of width and height of downscaled images - down_widths, down_heights
	* number of levels - num_levels
	*/
	int wd_height = WIN_SIZE, wd_width = WIN_SIZE;
	//scale_factor = 1;
	//minNeighbors = 1;
	unsigned int * num_stages = new unsigned int;
	Stage * stages_gpu = loadParametersToGPU(num_stages);

	int counter = 0;

#if GPUII == 1
	for (int i = 0; i < ii_depth; i++) {
#elif GPUII == 0
	for (int i = 0;;i++){
#endif
		/* size of the image scaled up */
		MySize winSize = { myRound_cpp(wd_width*scale_factor), myRound_cpp(wd_height*scale_factor) };
		/* size of the image scaled down (from bigger to smaller) */
		MySize sz = { (img->width / scale_factor), (img->height / scale_factor) };
		/* difference between sizes of the scaled image and the original detection window */
		MySize sz1 = { sz.width - wd_width, sz.height - wd_height };
		/* if the actual scaled image is smaller than the original detection window, break */

		//if (sz1.width < 0 || sz1.height < 0)
		//	break;

#if GPUII == 0

		if (sz1.width < 0 || sz1.height < 0)
			break;

		setImage(sz.width, sz.height, img1);
		setSumImage(sz.width, sz.height, sum1);
		setSumImage(sz.width, sz.height, sqsum1);
		nearestNeighbor_cpp(img, img1);
		integralImages_cpp(img1, sum1, sqsum1);
#endif

		//std::ofstream myfile;
		//myfile.open("integral_sum_sqsum_cu.txt");
		//if (myfile.is_open()) {
		//	for (int count = 0; count < sum1->width * sum1->height; count++) {
		//		//printf("%d ", sum1->data[count]);
		//		myfile << sum1->data[count] << " ";
		//	}
		//	myfile << "\nSQSUM\n";
		//	for (int count = 0; count < sum1->width * sum1->height; count++) {
		//		myfile << sqsum1->data[count] << " ";
		//	}
		//	myfile.close();
		//}
		//else std::cout << "Unable to open file";

#if GPUII == 1
		unsigned int * d_sum1 = ii_gpu[i];
		unsigned int * d_sqsum1 = sqii_gpu[i];
		unsigned int down_h = ii_sizes[2 * i + 1];//sz.height;
		unsigned int down_w = ii_sizes[2 * i];// sz.width;
#elif GPUII == 0
		unsigned int* d_sum1;
		unsigned int* d_sqsum1;
		CHECK(cudaMalloc((void**)&d_sum1, sum1->width * sum1->height * sizeof(int)));
		CHECK(cudaMalloc((void**)&d_sqsum1, sqsum1->width * sqsum1->height * sizeof(int)));
		CHECK(cudaMemcpy(d_sum1, sum1->data, sum1->width * sum1->height * sizeof(int), cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_sqsum1, sqsum1->data, sqsum1->width * sqsum1->height * sizeof(int), cudaMemcpyHostToDevice));

		unsigned int down_h = sz.height;
		unsigned int down_w = sz.width;
#endif

		unsigned int num_wd_col = down_h - wd_height + 1;  // number of windows on a column
		unsigned int num_wd_row = down_w - wd_width + 1;  // number of windows on a row
		//printf("sum dim: %d * %d, activation dim: %d * %d\n", down_h, down_w, num_wd_col, num_wd_row);

		// create activation map
		unsigned int act_size = num_wd_col * num_wd_row * sizeof(bool);
		bool *h_activation = (bool *)malloc(act_size);
		bool *d_activation;
		CHECK(cudaMalloc((void**)&d_activation, act_size));

		// run sliding window kernel
		dim3 blockDim(1024);
		dim3 gridDim((num_wd_row + blockDim.x - 1) / blockDim.x, down_h - wd_height + 1);
		//dim3 blockDim(1);
		//dim3 gridDim(1);
		//printf("grid dim: %d %d\n", gridDim.x, gridDim.y);

		//unsigned int params_size = (2913 * 18 + 25) * sizeof(int);
		//int *h_params = (int *)malloc(params_size);
		//int *d_params;
		//CHECK(cudaMalloc((void**)&d_params, params_size));

		slide_window << <gridDim, blockDim >> > (d_sum1, d_sqsum1, d_activation, wd_width, wd_height, down_w, down_h, stages_gpu);
		CHECK(cudaDeviceSynchronize());

		//// copy params back
		//CHECK(cudaMemcpy(h_params, d_params, params_size, cudaMemcpyDeviceToHost));
		//myfile.open("params.txt");
		//if (myfile.is_open()) {
		//	for (int count = 0; count < 2913 * 18 + 25; count++) {
		//		//printf("%d ", sum1->data[count]);
		//		myfile << h_params[count] << "\n";
		//	}
		//	myfile.close();
		//}
		//else std::cout << "Unable to open file";

		// copy activation map back
		CHECK(cudaMemcpy(h_activation, d_activation, act_size, cudaMemcpyDeviceToHost));

		// add activations to vector
		
		for (unsigned int j = 0; j < num_wd_col * num_wd_row; j++) {
			if (h_activation[j]) {
				unsigned int y = j / num_wd_row;
				unsigned int x = j % num_wd_row;
				MyRect r = { myRound_cpp(x * scale_factor), myRound_cpp(y * scale_factor), wd_width * scale_factor, wd_height * scale_factor };
				allCandidates.push_back(r);
				counter++;
			}
		}
		// increment scale factor
		scale_factor *= 1.2;		
	}
	//printf("num detected: %d, ", counter);

	// sort, clean and organize the labeled windows
	if (minNeighbors != 0) {
		groupRectangles(allCandidates, minNeighbors, GROUP_EPS);
	}
	
}




/*****************************************************
* Compute the integral image (and squared integral)
* Integral image helps quickly sum up an area.
* More info:
* http://en.wikipedia.org/wiki/Summed_area_table
****************************************************/
void integralImages_cpp(MyImage *src, MyIntImage *sum, MyIntImage *sqsum)
{
	int x, y, s, sq, t, tq;
	unsigned char it;
	int height = src->height;
	int width = src->width;
	unsigned char *data = src->data;
	int * sumData = sum->data;
	int * sqsumData = sqsum->data;
	for (y = 0; y < height; y++)
	{
		s = 0;
		sq = 0;
		/* loop over the number of columns */
		for (x = 0; x < width; x++)
		{
			it = data[y*width + x];
			/* sum of the current row (integer)*/
			s += it;
			sq += it*it;

			t = s;
			tq = sq;
			if (y != 0)
			{
				t += sumData[(y - 1)*width + x];
				tq += sqsumData[(y - 1)*width + x];
			}
			sumData[y*width + x] = t;
			sqsumData[y*width + x] = tq;
		}
	}
}

/***********************************************************
* This function downsample an image using nearest neighbor
* It is used to build the image pyramid
**********************************************************/
void nearestNeighbor_cpp(MyImage *src, MyImage *dst)
{

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

	unsigned char* src_data = src->data;
	unsigned char* dst_data = dst->data;


	int x_ratio = (int)((w1 << 16) / w2) + 1;
	int y_ratio = (int)((h1 << 16) / h2) + 1;

	for (i = 0; i<h2; i++)
	{
		t = dst_data + i*w2;
		y = ((i*y_ratio) >> 16);
		p = src_data + y*w1;
		rat = 0;
		for (j = 0; j<w2; j++)
		{
			x = (rat >> 16);
			*t++ = p[x];
			rat += x_ratio;
		}
	}
}

