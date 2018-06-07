#pragma once
#include <cuda_runtime.h>

class Rectangle {
	public:
		unsigned int x;
		unsigned int y;
		unsigned int height;
		unsigned int width;
		int weight;

		__device__ Rectangle() {}
		__device__ Rectangle(unsigned int x, unsigned int y, unsigned int height, unsigned int width, int weight) {
			this->x = x;
			this->y = y;
			this->height = height;
			this->width = width;
			this->weight = weight;
		}

		template <class T>
		__device__ T getSummedArea(T * arr, unsigned int arr_width) {
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
		unsigned char num_rect;

		__device__ Filter() {}
		__device__ Filter(	unsigned int x1,
				unsigned int y1,
				unsigned int width1,
				unsigned int height1,
				int weight1,
				unsigned int x2,
				unsigned int y2,
				unsigned int width2,
				unsigned int height2,
				int weight2,
				int threshold,
				int alpha1,
				int alpha2) {
			rect[0] = Rectangle(x1, y1, height1, width1, weight1);
			rect[1] = Rectangle(x2, y2, height2, width2, weight2);
			num_rect = 2;
			this->threshold = threshold;
			this->alpha1 = alpha1;
			this->alpha2 = alpha2;
		}

		__device__ Filter(	unsigned int x1,
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
			num_rect = 3;
			this->threshold = threshold;
			this->alpha1 = alpha1;
			this->alpha2 = alpha2;
		}

		template <class T>
		__device__ T getValue(T * arr, unsigned int arr_width) {
			T area = 0;
			for (int i = 0; i < num_rect; i++) {
				area += rect[i].weight*rect[i].getSummedArea(arr, arr_width);
			}
			return (area >= threshold) ? alpha2 : alpha1;
		}
};
