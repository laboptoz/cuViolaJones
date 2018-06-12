#pragma once
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "cpuViolaJones.hpp"
#include "paths.hpp"
#include "gpuViolaJones.cuh"
#include "load_images.hpp"
#include "haar.cuh"
#include <stdlib.h>
#include "image.h"
#include "macros.hpp"


using namespace std;
using namespace cv;

void run_vj_gpu(Mat gray_face);

int main(int argc, char** argv )
{	
	switch(MODE) {
		case 0: {
			Mat image = imread(FACE_PATH, 1);
			if (!image.data) {
				printf("No image data \n");
				return -1;
			}

			Mat gray_face;
			cvtColor(image, gray_face, CV_BGR2GRAY);
			run_vj_gpu(gray_face);
			break;
		}

		case 1: {
			// Load test images
			int *numImgs = new int;
			Image *imgs = loadData(LABEL_PATH, IMAGE_PATH, numImgs);
			testCpuViolaJones(imgs, *numImgs, DISPLAY);
			char c; printf("Press ENTER to continue...\n"); cin.get(c);
			testGpuViolaJones(imgs, *numImgs, DISPLAY);
			break;
		}

		case 2: {
			webcamTest();
			break;
		}
	}

    return 0;
}


void run_vj_gpu(Mat gray_face) {

	MyImage imageObj;
	MyImage *image = &imageObj;
	image->data = gray_face.data;
	image->width = gray_face.cols;
	image->height = gray_face.rows;
	image->maxgrey = 255;

	std::vector<MyRect> result;
	detect_faces(image->width, image->height, result, image, SCALING, MIN_NEIGH);
	cout << "Size: " << result.size() << endl;
	for (int i = 0; i < result.size(); i++) {
		MyRect r = result[i];
		drawRectangle(image, r);
	}

	imshow("CPU Result", gray_face);
	waitKey(0);

}