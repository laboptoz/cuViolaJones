#pragma once
#include <cuda_runtime.h>

class Filter {
public:
	unsigned int x1;
	unsigned int y1;
	unsigned int width1;
	unsigned int height1;
	int weight1;
	unsigned int x2;
	unsigned int y2;
	unsigned int width2;
	unsigned int height2;
	int weight2;
	unsigned int x3;
	unsigned int y3;
	unsigned int width3;
	unsigned int height3;
	int weight3;
	int threshold;
	int alpha1;
	int alpha2;

	template <typename T>
	__inline__ __device__ T getSummedArea(T * arr, unsigned int arr_width, unsigned int x, unsigned int y, unsigned int width, unsigned int height) {

		//printf("-- s: %d, %d, %d, %d\n", *(arr + x + y*arr_width), *(arr + x + width + y*arr_width),
		//	*(arr + x + (y+height)*arr_width), *(arr + x + width + (y + height)*arr_width));

		return *(arr + (x + width) + (y + height)*arr_width) +
			*(arr + x + y*arr_width) -
			*(arr + x + width + y*arr_width) -
			*(arr + x + (y + height)*arr_width);
	}

	__host__ __device__ Filter() {}

	//Default 3 rectangle filter constructor (2 rectangle uses 0s in place of rectangle 3 values)
	__host__ __device__ Filter(unsigned int x1,
		unsigned int y1,
		unsigned int width1,
		unsigned int height1,
		int weight1,
		unsigned int x2,
		unsigned int y2,
		unsigned int width2,
		unsigned int height2,
		int weight2,
		unsigned int x3,
		unsigned int y3,
		unsigned int width3,
		unsigned int height3,
		int weight3,
		int threshold,
		int alpha1,
		int alpha2) {

		this->x1 = x1;
		this->y1 = y1;
		this->width1 = width1;
		this->height1 = height1;
		this->weight1 = weight1;
		this->x2 = x2;
		this->y2 = y2;
		this->width2 = width2;
		this->height2 = height2;
		this->weight2 = weight2;
		this->x3 = x3;
		this->y3 = y3;
		this->width3 = width3;
		this->height3 = height3;
		this->weight3 = weight3;
		this->threshold = threshold;
		this->alpha1 = alpha1;
		this->alpha2 = alpha2;
	}

	template <typename T>
	__host__ __device__ T getValue(T * arr, int norm_factor, unsigned int arr_width) {
		int s1 = getSummedArea<T>(arr, arr_width, x1, y1, width1, height1);
		int a1 = weight1 * s1;
		int s2 = getSummedArea<T>(arr, arr_width, x2, y2, width2, height2);
		int a2 = weight2 * s2;
		int s3 = 0;
		int a3 = 0;
		if (weight3 != 0) {
			s3 = getSummedArea<T>(arr, arr_width, x3, y3, width3, height3);
			a3 = weight3 * s3;
		}
		int area = a1 + a2 + a3;
		//printf("\n\n\n");
		//printf("--- filter area %d thresh %d ", area, norm_factor*threshold);
		//printf("-- thresh %d, norm %d\n", threshold, norm_factor);
		//printf("-- w1 %d s1 %d, w2 %d s2 %d, w3 %d s3 %d\n", weight1, s1, weight2, s2, weight3, s3);


		//Return alpha2 if filter passes, alpha1 if it does not
		return (area >= norm_factor*threshold)*alpha2 + (area < norm_factor*threshold)*alpha1;
	}
};

class Stage {
public:
	Filter * filters = nullptr;
	unsigned int num_filters = 0;
	int threshold = 0;

	__host__ __device__ Stage() {}
	__host__ __device__ Stage(int threshold, const unsigned int num_filters) {
		this->threshold = threshold;
		this->num_filters = num_filters;
	}


	//Returns true if the stage passes the threshold, false otherwise
	template <typename T>
	__device__ bool getValue(int * arr, float norm_factor, unsigned int arr_width) {
		int activations = 0;

		//Get the activation from each filter
		for (int i = 0; i < num_filters; i++) {
			//printf("filter %d ", i);
			activations += filters[i].getValue<int>(arr, norm_factor, arr_width);
			//printf(" %d ", activations);
			//break;

		}
		//printf("%d\n", activations);
		return (activations >= threshold);
	}

};