#include <stdio.h>
#include <iostream>
#include "HaarCascade.h"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;

int main(int argc, char** argv){
	unsigned int height = 512;
	unsigned int width = 1024;

	//GENERATE HAAR CLASSIFIER FROM OPENCV FILE
	cv::CascadeClassifier classifierOpenCV;
	classifierOpenCV.load("./data/haarcascade_frontalface_default.xml");
	void * test = cvLoad("./data/haarcascade_frontalface_default.xml");
	CvHaarClassifierCascade * haarCascade = (CvHaarClassifierCascade *) cvLoad("./data/haarcascade_frontalface_default.xml");
	HaarCascade hc_cpu = HaarCascade(haarCascade, height, width);


	//TEMPORARILY USE OPENCV INTEGRAL IMAGE
	//TODO: CHANGE THIS SECTION
	cv::Mat image = cv::Mat(height, width, CV_8UC1);
	for (int i = 0; i < 1024 * 512; i++) {
		image.data[i] = 1;
	}

	cv::Mat sum = cv::Mat(height+1, width+1, CV_32FC1);
	cv::Mat sqsum = cv::Mat(height + 1, width + 1, CV_32FC1);

	cv::integral(image, sum, sqsum);

	//LOAD DATA TO GPU
	HaarCascade hc_gpu = HaarCascade(hc_cpu);
	hc_cpu.copyToGPU(&hc_gpu);
	float *gpu_sum;
	double * gpu_sqsum;
	CHECK(cudaMalloc(&gpu_sum, sizeof(float)*(height + 1)*(width + 1)));
	CHECK(cudaMalloc(&gpu_sqsum, sizeof(double)*(height + 1)*(width + 1)));
	CHECK(cudaMemcpy(gpu_sum, sum.data, sizeof(float)*(height + 1)*(width + 1), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(gpu_sqsum, sqsum.data, sizeof(double)*(height + 1)*(width + 1), cudaMemcpyHostToDevice));
	

	//ALLOCATE SPACES FOR DETECTED FACES
	Rect * faces, faces_gpu;
	faces = (Rect *)malloc(width*height * sizeof(Rect));
	memset(faces, 0, width*height * sizeof(Rect));

	CHECK(cudaMalloc(&faces_gpu, width*height * sizeof(Rect)));
	CHECK(cudaMemcpy(faces_gpu, faces, width*height * sizeof(Rect), cudaMemcpyHostToDevice));

	//Determine image scales
	//TODO: CHANGE THIS SO THAT IT DOESN'T LOOK IDENTICAL
	double curr_scale = 1.0f;
	float scale_mult = 1.2f;
	std::vector<double> scale;

	while (curr_scale*hc_cpu.obj_width < width - 10 &&
		curr_scale*hc_cpu.obj_height < height - 10) {
		scale.push_back(curr_scale);
		curr_scale *= scale_mult;
	}

	for (int i = 0; i < scale.size(); i++) {
		//COPY SCALED FILTERS TO DEVICE
		hc_cpu.scaleFilters(scale[i]);
		CHECK(cudaMemcpy(hc_gpu.scaled_filters, hc_cpu.scaled_filters, hc_cpu.num_filters * sizeof(Filter), cudaMemcpyHostToDevice));

		hc_gpu.

	}

}
