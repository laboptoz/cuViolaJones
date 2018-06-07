#pragma once
#include <cuda_runtime.h>

class Rectangle {
	public:
		unsigned int x;
		unsigned int y;
		unsigned int height;
		unsigned int width;
		int weight;

		__host__ __device__ Rectangle() {}
		__host__ __device__ Rectangle(unsigned int x, unsigned int y, unsigned int height, unsigned int width, int weight) {
			this->x = x;
			this->y = y;
			this->height = height;
			this->width = width;
			this->weight = weight;
		}

		template <typename T>
		__host__ __device__ T getSummedArea(T * arr, unsigned int arr_width) {
			return *(arr + (x + width - 1) + (y + height - 1)*arr_width) +
				*(arr + x + y*arr_width) -
				*(arr + x + width - 1 + y*arr_width) -
				*(arr + x + (y + height - 1)*arr_width);
		}

};

class Filter {
	public: 
		Rectangle * rect = new Rectangle[3];
		int threshold;
		int alpha1, alpha2;

		__host__ __device__ Filter() {}

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

			rect[0] = Rectangle(x1, y1, height1, width1, weight1);
			rect[1] = Rectangle(x2, y2, height2, width2, weight2);
			rect[2] = Rectangle(x3, y3, height3, width3, weight3);
			this->threshold = threshold;
			this->alpha1 = alpha1;
			this->alpha2 = alpha2;
		}

		template <typename T>
		__host__ __device__ T getValue(T * arr, unsigned int arr_width) {
			T area = 0;
			#pragma unroll
			for (int i = 0; i < 3; i++) {
				area += rect[i].weight*rect[i].getSummedArea(arr, arr_width);
			}
			return (area >= threshold)*alpha2 + (area < threshold)*alpha1;
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

		template <typename T>
		__device__ bool getValue(T * arr, unsigned int arr_width) {
			T activations = 0;
			for (int i = 0; i < num_filt; i++) {
				activations += filters[i].getValue(arr, arr_width);
			}
			return (activations >= 0.4*threshold);
		}

};