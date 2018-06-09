#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "cpuViolaJones.hpp"
#include "paths.hpp"
#include "gpuViolaJones.cuh"
#include "load_images.hpp"

using namespace std;


int main(int argc, char** argv)
{
	cv::Mat image;
	image = cv::imread(FACE_PATH, 1);
	cv::String face_cascade_path = CASCADE_PATH;
	if (!image.data)
	{
		printf("No image data \n");
		return -1;
	}

	cv::Mat gray_face;
	cv::cvtColor(image, gray_face, CV_BGR2GRAY);
	unsigned int height = 0;
	unsigned int width = 0;
	height = gray_face.rows;
	width = gray_face.cols;
	unsigned int large = max(height, width);
	if (height > 1024 || width > 1024) {
		float large_scale = 1024.0 / large;
		cv::resize(gray_face, gray_face, Size(), large_scale, large_scale);
		height = gray_face.rows;
		width = gray_face.cols;
	}
	imshow("gray", gray_face);
	unsigned char * face;
	if (gray_face.isContinuous()) {
		face = gray_face.data;
		height = gray_face.rows;
		width = gray_face.cols;
	}
	else {
		fprintf(stderr, "Stop\n");
	}

	// Load test images
	printf("Loading image set\n");
	int *numImgs = new int;

	//TEST CODE
	unsigned int width1 = 100;
	unsigned int height1 = 100;
	unsigned char * input = new unsigned char[width1 * height1];
	for (int i = 0; i < width1 * height1; i++) {
		input[i] = i + 1;
	}
	unsigned int min_size = 24;
	float scale = 1.2;
	//END TEST CODE
	unsigned char * result = gpuViolaJones(face, width, height, 24, 1.2);

	Mat result_img = Mat(height, width, CV_8U, result);
	imshow("test", result_img);

	cpuViolaJones(image, face_cascade_path);
	waitKey(0);

	return 0;
}
