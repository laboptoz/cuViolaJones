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
			gpuSingleDetection(gray_face);
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

