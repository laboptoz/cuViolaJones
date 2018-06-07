#include <cuda_runtime.h>
//#include "cuIntegralImage.cuh"
//#include "cuNNII.h"

#include "haar.h"
#include "image.h"
#include <stdio.h>
#include "stdio-wrapper.h"
#include "paths.hpp"

#include <cuda.h>


/* compute integral images */
void integralImages(MyImage *src, MyIntImage *sum, MyIntImage *sqsum);

/* scale down the image */
void ScaleImage_Invoker(myCascade* _cascade, float _factor, int sum_row, int sum_col, std::vector<MyRect>& _vec);

/* compute scaled image */
void nearestNeighbor(MyImage *src, MyImage *dst);

/* rounding function */
inline  int  myRound(float value)
{
	return (int)(value + (value >= 0 ? 0.5 : -0.5));
}

/*******************************************************
* Function: detectObjects
* Description: It calls all the major steps
******************************************************/


__device__ float run_filter(float *window, Filter *ft, int img_width, float norm_factor) {
	// window: pointer to top left corner of window

	return ft->getValue(window, img_width, norm_factor)
}


__device__ int operation(int idx, int img_width, float norm_factor) {
	// idx: top left corner index of the window

	/*
	* TODO: 
	* assume i have an array of number of filters per stage - filter_counts
	* filter class - Filter
	* array of pointers to filters - filters
	* sum integral image [0] pointer - sum_int
	* stage threshold - 
	*
	* UPDATE filter function to take argument of variance normalization factor
	*/ 
	const int num_stages = 25;

	int num_filters;
	int stage_sum;

	for (int i = 0; i < num_stages; i++) {
		num_filters = filter_counts[i];
		stage_sum = 0;
		for (int j = 0; j < num_filters; j++) {
			Filter *ft = filters[i];
			unsigned int offset = idx * sizeof(float);
			stage_sum += run_filter(sum_int + offset, ft, img_width, float norm_factor);
		}
		if (stage_sum < 0.4 * stage_thresh)
			return 0
	}
	return 1;
}

__global__ void slide_window(
	int *d_activation;
	unsigned int wd_width, unsigned int wd_height, 
	unsigned int img_width, unsigned int img_height) {
	//read filter parameters

	// get four corners
	unsigned int topLeft_idx = blockIdx.y * img_width + blockIdx.x * blockDim.x + threadIdx.x;
	//float val0 = d_image[topLeft_idx];
	//float val1 = d_image[topLeft_idx + wd_width - 1];
	//float val2 = d_image[topLeft_idx + (wd_height - 1) * img_width];
	//float val3 = d_image[topLeft_idx + (wd_height - 1) * img_width + wd_width - 1];

	int result = operation(topLeft_idx, img_width);
	result *= ((topLeft_idx + 1) / img_width == blockIdx.y);  // to differentiate if goes outside of image

	unsigned int dest_idx = blockIdx.y * (img_width - wd_width + 1) + blockIdx.x * blockDim.x + threadIdx.x;
	d_activation[dest_idx] = result;
}

void detect_faces(unsigned int img_width, unsigned int img_height, std::vector<MyRect> &allCandidates) {

	/*
	* TODO:
	* assume we have array of width and height of downscaled images - down_widths, down_heights
	* number of levels - num_levels
	*/
	int wd_height = 24, wd_width = 24;
	float scale_factor = 1;
	int minNeighbors = 1;

	for (int i = 0; i < num_level; i++) {
		unsigned int down_h = down_heights[i], down_w = down_widths[i];
		unsigned int num_wd_row = down_w - wd_width + 1;  // number of windows on a row
		unsigned int num_wd_col = down_h - wd_height + 1;  // number of windows on a column
		unsigned int act_size = num_wd_col * num_wd_row * sizeof(int)
		int *h_activation = (int *)malloc(act_size);
		int *d_activation;
		CHECK(cudaMalloc(d_activation, act_size));

		dim3 blockDim(32);
		dim3 gridDim((num_wd_row + blockDim.x - 1) / blockDim.x, down_h - wd_height + 1);
		printf("grid dim: %d %d\n", gridDim.x, gridDim.y);
		slide_window << <gridDim, blockDim >> > (d_activation, wd_width, wd_height, down_w, down_h);

		CHECK(cudaMemcpy(h_activation, d_activation, act_size, cudaMemcpyDeviceToHost));

		// add activations to vector
		for (unsigned int j = 0; j < num_wd_col * num_wd_row; j++) {
			if (h_activation[j] == 1) {
				unsigned int y = j / num_wd_row;
				unsigned int x = j % num_wd_row;
				MyRect r = { myRound(x * scale_factor), myRound(y * scale_factor), wd_width * scale_factor, wd_height * scale_factor };
				allCandidates->push_back(r);
			}
		}
		// increment scale factor
		scale_factor *= 1.2;		
	}

	// sort, clean and organize the labeled windows
	if (minNeighbors != 0) {
		groupRectangles(allCandidates, minNeighbors, GROUP_EPS);
	}
	
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
unsigned int int_sqrt(unsigned int value)
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


/*****************************************************
* Compute the integral image (and squared integral)
* Integral image helps quickly sum up an area.
* More info:
* http://en.wikipedia.org/wiki/Summed_area_table
****************************************************/
void integralImages(MyImage *src, MyIntImage *sum, MyIntImage *sqsum)
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
void nearestNeighbor(MyImage *src, MyImage *dst)
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

void readTextClassifier()//(myCascade * cascade)
{
	/*number of stages of the cascade classifier*/
	int stages;
	/*total number of weak classifiers (one node each)*/
	int total_nodes = 0;
	int i, j, k, l;
	char mystring[12];
	int r_index = 0;
	int w_index = 0;
	int tree_index = 0;
	FILE *finfo = fopen(INFO_PATH, "r");

	/**************************************************
	/* how many stages are in the cascaded filter?
	/* the first line of info.txt is the number of stages
	/* (in the 5kk73 example, there are 25 stages)
	**************************************************/
	if (fgets(mystring, 12, finfo) != NULL) {
		stages = atoi(mystring);
	}
	i = 0;

	stages_array = (int *)malloc(sizeof(int)*stages);

	/**************************************************
	* how many filters in each stage?
	* They are specified in info.txt,
	* starting from second line.
	* (in the 5kk73 example, from line 2 to line 26)
	*************************************************/
	while (fgets(mystring, 12, finfo) != NULL)
	{
		stages_array[i] = atoi(mystring);
		total_nodes += stages_array[i];
		i++;
	}
	fclose(finfo);


	/* TODO: use matrices where appropriate */
	/***********************************************
	* Allocate a lot of array structures
	* Note that, to increase parallelism,
	* some arrays need to be splitted or duplicated
	**********************************************/
	rectangles_array = (int *)malloc(sizeof(int)*total_nodes * 12);
	scaled_rectangles_array = (int **)malloc(sizeof(int*)*total_nodes * 12);
	weights_array = (int *)malloc(sizeof(int)*total_nodes * 3);
	alpha1_array = (int*)malloc(sizeof(int)*total_nodes);
	alpha2_array = (int*)malloc(sizeof(int)*total_nodes);
	tree_thresh_array = (int*)malloc(sizeof(int)*total_nodes);
	stages_thresh_array = (int*)malloc(sizeof(int)*stages);
	FILE *fp = fopen(CLASS_PATH, "r");

	/******************************************
	* Read the filter parameters in class.txt
	*
	* Each stage of the cascaded filter has:
	* 18 parameter per filter x tilter per stage
	* + 1 threshold per stage
	*
	* For example, in 5kk73,
	* the first stage has 9 filters,
	* the first stage is specified using
	* 18 * 9 + 1 = 163 parameters
	* They are line 1 to 163 of class.txt
	*
	* The 18 parameters for each filter are:
	* 1 to 4: coordinates of rectangle 1
	* 5: weight of rectangle 1
	* 6 to 9: coordinates of rectangle 2
	* 10: weight of rectangle 2
	* 11 to 14: coordinates of rectangle 3
	* 15: weight of rectangle 3
	* 16: threshold of the filter
	* 17: alpha 1 of the filter
	* 18: alpha 2 of the filter
	******************************************/

	/* loop over n of stages */
	for (i = 0; i < stages; i++)
	{    /* loop over n of trees */
		for (j = 0; j < stages_array[i]; j++)
		{	/* loop over n of rectangular features */
			for (k = 0; k < 3; k++)
			{	/* loop over the n of vertices */
				for (l = 0; l <4; l++)
				{
					if (fgets(mystring, 12, fp) != NULL)
						rectangles_array[r_index] = atoi(mystring);
					else
						break;
					r_index++;
				} /* end of l loop */
				if (fgets(mystring, 12, fp) != NULL)
				{
					weights_array[w_index] = atoi(mystring);
					/* Shift value to avoid overflow in the haar evaluation */
					/*TODO: make more general */
					/*weights_array[w_index]>>=8; */
				}
				else
					break;
				w_index++;
			} /* end of k loop */
			if (fgets(mystring, 12, fp) != NULL)
				tree_thresh_array[tree_index] = atoi(mystring);
			else
				break;
			if (fgets(mystring, 12, fp) != NULL)
				alpha1_array[tree_index] = atoi(mystring);
			else
				break;
			if (fgets(mystring, 12, fp) != NULL)
				alpha2_array[tree_index] = atoi(mystring);
			else
				break;
			tree_index++;
			if (j == stages_array[i] - 1)
			{
				if (fgets(mystring, 12, fp) != NULL)
					stages_thresh_array[i] = atoi(mystring);
				else
					break;
			}
		} /* end of j loop */
	} /* end of i loop */
	fclose(fp);
}


void releaseTextClassifier()
{
	free(stages_array);
	free(rectangles_array);
	free(scaled_rectangles_array);
	free(weights_array);
	free(tree_thresh_array);
	free(alpha1_array);
	free(alpha2_array);
	free(stages_thresh_array);
}