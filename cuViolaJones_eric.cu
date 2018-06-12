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
		// Detects a single image
		case 0: {
			Mat image = imread(SWIM, 1);
			if (!image.data) {
				printf("No image data \n");
				return -1;
			}
			gpuSingleDetection(image);
			break;
		}

		// Run metric test for CPU and GPU
		case 1: {
			// Load test images
			int *numImgs = new int;
			Image *imgs = loadData(LABEL_PATH, IMAGE_PATH, numImgs);
			testCpuViolaJones(imgs, *numImgs, DISPLAY);
			char c; printf("Press ENTER to continue...\n"); cin.get(c);
			imgs = loadData(LABEL_PATH, IMAGE_PATH, numImgs);
			testGpuViolaJones(imgs, *numImgs, DISPLAY);
			break;
		}

		// Webcam mode
		case 2: {
			webcamTest();
			break;
		}
	}

    return 0;
}

