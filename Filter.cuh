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
			return *(arr + (x + width - 1) + (y + height - 1)*arr_width) +
				*(arr + x + y*arr_width) -
				*(arr + x + width - 1 + y*arr_width) -
				*(arr + x + (y + height - 1)*arr_width);
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
		__host__ __device__ T getValue(T * arr, float norm_factor, unsigned int arr_width) {
			if (threadIdx.x + blockIdx.y == 0)
				printf("test\n");
			T area = weight1*getSummedArea(arr, arr_width, x1, y1, width1, height1)
				+ weight2*getSummedArea(arr, arr_width, x2, y2, width2, height2)
				+ weight3*getSummedArea(arr, arr_width, x3, y3, width3, height3);
			if (threadIdx.x + blockIdx.y == 0)
				printf("test2\n");
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
		__device__ bool getValue(T * arr, float norm_factor, unsigned int arr_width) {
			T activations = 0;

			//Get the activation from each filter
			for (int i = 0; i < num_filters; i++) {
				activations += filters[i].getValue(arr, norm_factor, arr_width);
			}
			return (activations >= 0.4*threshold);
		}

};