#pragma once
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include "cudaErrorCheck.h"

class Rect {
	float x;
	float y;
	float width;
	float height;
	__device__ __host__ Rect() {}
	__device__ __host__ Rect(float x, float y, float w, float h) {
		this->x = x;
		this->y = y;
		width = w;
		height = h;
	}
};

class Filter{
public:
	float threshold;
	float alpha0;
	float alpha1;
	unsigned int x0, y0, w0, h0, a0;
	unsigned int x1, y1, w1, h1, a1;
	unsigned int x2, y2, w2, h2, a2;
};

class Stage {
public:
	float threshold;
	unsigned int num_filters;
	unsigned int filter_offset;
};

class HaarCascade {
public:
	unsigned int num_stages;
	unsigned int num_filters = 0;
	unsigned int win_height;
	unsigned int win_width;
	unsigned int obj_height;
	unsigned int obj_width;
	unsigned int im_height;
	unsigned int im_width;
	unsigned int scale_height;
	unsigned int scale_width;

	float scale;

	Stage * stages;
	Filter * filters;
	Filter * scaled_filters;

	HaarCascade() {}
	HaarCascade(CvHaarClassifierCascade * cascade, unsigned int height, unsigned int width) {
		num_stages = cascade->count;
		win_height = cascade->real_window_size.height;
		win_width = cascade->real_window_size.width;
		obj_height = cascade->orig_window_size.height;
		obj_width = cascade->orig_window_size.width;
		scale = cascade->scale;
		im_height = height;
		im_width = width;

		for (int i = 0; i < num_stages; i++) {
			num_filters += cascade->stage_classifier[i].count;
		}

		stages = new Stage[num_stages];
		filters = new Filter[num_filters];
		scaled_filters = new Filter[num_filters];

		unsigned int filter_offset = 0;
		for (int i = 0; i < num_stages; i++) {
			CvHaarStageClassifier cvs = cascade->stage_classifier[i];
			Stage s;
			s.threshold = cvs.threshold;
			s.num_filters = cvs.count;
			s.filter_offset = filter_offset;

			for (int j = 0; j < s.num_filters; j++) {
				CvHaarClassifier cvfilt = cvs.classifier[j];
				Filter filt;
				filt.threshold = cvfilt.threshold[0];
				filt.alpha0 = cvfilt.alpha[0];
				filt.alpha1 = cvfilt.alpha[1];

				CvHaarFeature cvfeat = cvfilt.haar_feature[0];
				filt.x0 = cvfeat.rect[0].r.x;
				filt.y0 = cvfeat.rect[0].r.y;
				filt.w0 = cvfeat.rect[0].r.width;
				filt.h0 = cvfeat.rect[0].r.height;
				filt.a0 = cvfeat.rect[0].weight;

				filt.x1 = cvfeat.rect[1].r.x;
				filt.y1 = cvfeat.rect[1].r.y;
				filt.w1 = cvfeat.rect[1].r.width;
				filt.h1 = cvfeat.rect[1].r.height;
				filt.a1 = cvfeat.rect[1].weight;

				filt.x2 = cvfeat.rect[2].r.x;
				filt.y2 = cvfeat.rect[2].r.y;
				filt.w2 = cvfeat.rect[2].r.width;
				filt.h2 = cvfeat.rect[2].r.height;
				filt.a2 = cvfeat.rect[2].weight;

				filters[filter_offset] = filt;
				scaled_filters[filter_offset++] = filt;
			} //END FILTER LOOP
			stages[i] = s;
		} //END STAGE LOOP

	} //END loadToGPU()

	//Copy constructor
	HaarCascade(const HaarCascade &hc) {
		num_stages = hc.num_stages;
		num_filters = hc.num_filters;
		win_height = hc.win_height;
		win_width = hc.win_width;
		obj_height = hc.obj_height;
		obj_width = hc.obj_width;
		im_height = hc.im_height;
		im_width = hc.im_width;
		scale = hc.scale;

		stages = (Stage*)malloc(num_stages * sizeof(Stage));
		filters = (Filter *)malloc(num_filters * sizeof(Filter));
		scaled_filters = (Filter *)malloc(num_filters * sizeof(Filter));

		for (int i = 0; i < num_stages; i++) {
			stages[i] = hc.stages[i];
		}

		for (int i = 0; i < num_filters; i++) {
			filters[i] = hc.filters[i];
			scaled_filters[i] = hc.filters[i];
		}
	}

	void copyToGPU(HaarCascade * hc_gpu) {
		//COPY SELF TO GPU
		CHECK(cudaMalloc(&hc_gpu, sizeof(HaarCascade)));

		//ALLOCATE SPACE ON DEVICE
		Stage * stage_tmp = stages;
		Filter * filter_tmp = filters;
		Filter * filter_sc_tmp = scaled_filters;
		CHECK(cudaMalloc(&stages, sizeof(Stage)*num_stages));
		CHECK(cudaMalloc(&filters, sizeof(Filter)*num_filters));
		CHECK(cudaMalloc(&scaled_filters, sizeof(Filter)*num_filters));

		CHECK(cudaMemcpy(hc_gpu, this, sizeof(HaarCascade), cudaMemcpyHostToDevice));

		//COPY TO DEVICE
		CHECK(cudaMemcpy(filters, filter_tmp, sizeof(Filter)*num_filters, cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(scaled_filters, filter_sc_tmp, sizeof(Filter)*num_filters, cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(stages, stage_tmp, sizeof(Stage)*num_stages, cudaMemcpyHostToDevice));
	}

	void scaleFilters(float scale) {
		this->scale = scale;

		win_width = round(obj_width*1.2);
		win_height = round(obj_height*1.2);

		scale_width = round(im_width*win_width);
		scale_height = round(im_height*win_height);

		for (int i = 0; i < num_filters; i++) {
			scaled_filters[i].x0 = filters[i].x0 * scale;
			scaled_filters[i].y1 = filters[i].y0 * scale;
			scaled_filters[i].w0 = filters[i].w0 * scale;
			scaled_filters[i].h0 = filters[i].h0 * scale;

			scaled_filters[i].x1 = filters[i].x1 * scale;
			scaled_filters[i].y1 = filters[i].y1 * scale;
			scaled_filters[i].w1 = filters[i].w1 * scale;
			scaled_filters[i].h1 = filters[i].h1 * scale;

			if (scaled_filters[i].a2) {
				scaled_filters[i].x2 = filters[i].x2 * scale;
				scaled_filters[i].y2 = filters[i].y2 * scale;
				scaled_filters[i].w2 = filters[i].w2 * scale;
				scaled_filters[i].h2 = filters[i].h2 * scale;
			}
		}
	}
};
