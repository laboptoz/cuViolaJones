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
	//============================================
	//       Disables CPU multithreading
	//============================================
	if (!CPUMULTI) {
		setNumThreads(0);
	}

	switch(MODE) {
		//============================================
		//       Detects a single image
		//============================================
		case 0: {
			//Read image
			Mat image = imread(FACES, 1);

			//If no image loaded, throw error
			if (!image.data) {
				printf("No image data \n");
				return -1;
			}

			//Run single image detection
			gpuSingleDetection(image);
			break;
		}

		//============================================
		//       Run metric test for CPU and GPU
		//============================================
		case 1: {

			// Load test images
			int *numImgs = new int;
			Image *imgs;

			//============================================
			//       If CPU test option is enabled
			//============================================
			if (CPUTEST) {
				imgs = loadData(LABEL_PATH, IMAGE_PATH, numImgs);
				if (NUMIMGS != 0) {
					*numImgs = NUMIMGS;
				}
				testCpuViolaJones(imgs, *numImgs, DISPLAY);
				char c; printf("Press ENTER to continue...\n"); cin.get(c);
			}

			//============================================
			//       If GPU test option is enabled
			//============================================
			if (GPUTEST) {
				imgs = loadData(LABEL_PATH, IMAGE_PATH, numImgs);
				if (NUMIMGS != 0) {
					*numImgs = NUMIMGS;
				}
				testGpuViolaJones(imgs, *numImgs, DISPLAY);
			}
			break;
		}

		//============================================
		//       Webcam mode
		//============================================
		case 2: {
			webcamTest();
			break;
		}
	}

    return 0;
}

