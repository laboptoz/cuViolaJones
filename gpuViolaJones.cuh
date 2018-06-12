#pragma once
#include <stdio.h>
#include <cuda_runtime.h>
#include "cuNNII.h"
#include "Filter.cuh"
#include "haar.cuh"
#include "image.h"
#include "cuda_error_check.h"
#include "parameter_loader.h"

using namespace std;
using namespace cv;

__global__ void cascadeClassifier(Stage *, unsigned int, unsigned char *, unsigned int *, unsigned int *, unsigned int, unsigned int, unsigned int, unsigned int);

unsigned char * gpuViolaJones(unsigned char * img, unsigned int width, unsigned int height, unsigned int win_size, float scale) {
	
	printf("Loading cascade classifier information to GPU\n");
	//READ IN CASCADE CLASSIFIER INFORMATION
	unsigned int * num_stages = new unsigned int;
	Stage * stages_gpu = loadParametersToGPU(num_stages);

	printf("Generating image and squared image pyramids\n");
	//GENERATE THE INTEGRAL IMAGE PYRAMID
	unsigned int * iiPyramidSizes = nullptr;
	unsigned int iiPyramidDepth = 0;
	unsigned int ** iiPyramid_gpu = generateImagePyramid<false>(img, &iiPyramidSizes, &iiPyramidDepth, win_size, width, height, scale);

	//GENERATE THE SQUARED INTEGRAL IMAGE PYRAMID
	unsigned int * viiPyramidSizes = nullptr;
	unsigned int viiPyramidDepth = 0;
	unsigned int ** viiPyramid_gpu = generateImagePyramid<true>(img, &viiPyramidSizes, &viiPyramidDepth, win_size, width, height, scale);
	
	unsigned char * activationMask = (unsigned char *)malloc(sizeof(unsigned char)*iiPyramidSizes[0] * iiPyramidSizes[1]);
	unsigned char * activationMask_gpu;
	CHECK(cudaMalloc(&activationMask_gpu, sizeof(unsigned char)*iiPyramidSizes[0] * iiPyramidSizes[1]));

	//Loop through depth levels
	for (unsigned int i = 0; i < 1; i++) {
		printf("Depth: %u, Width: %u, Height: %u, Window Size: %u\n", i, iiPyramidSizes[i * 2], iiPyramidSizes[2 * i + 1], win_size);
		unsigned int threads = max(iiPyramidSizes[i * 2]-win_size, 1024);
		dim3 block = dim3(threads);
		dim3 grid = dim3((iiPyramidSizes[2*i]-win_size+1023)/1024, iiPyramidSizes[2*i+1]- win_size);

		cascadeClassifier<<<grid,block>>>(stages_gpu, *num_stages, activationMask_gpu, iiPyramid_gpu[i], viiPyramid_gpu[i], i, win_size, iiPyramidSizes[i * 2], iiPyramidSizes[i * 2 + 1]);
		CHECK(cudaDeviceSynchronize());

	}

	CHECK(cudaMemcpy(activationMask, activationMask_gpu, sizeof(unsigned char)*iiPyramidSizes[0] * iiPyramidSizes[1], cudaMemcpyDeviceToHost));

	return activationMask;
}
__global__ void cascadeClassifier(Stage * stages, unsigned int num_stages, unsigned char * activationMask, unsigned int * iiPyramid, unsigned int * viiPyramid, unsigned int depth, unsigned int window_size, unsigned int width, unsigned int height) {
	unsigned int offset = threadIdx.x + blockIdx.x*blockDim.x + blockIdx.y*width;
	float mean = 0;
	float norm_factor = 0;
	unsigned int pass = 0;
	norm_factor = viiPyramid[blockIdx.y*width + threadIdx.x]
				- viiPyramid[blockIdx.y*width + threadIdx.x + window_size]
				- viiPyramid[blockIdx.y*width + threadIdx.x + window_size]
				+ viiPyramid[blockIdx.y*width + threadIdx.x + window_size + window_size*width];
	mean	= iiPyramid[blockIdx.y*width + threadIdx.x]
			- iiPyramid[blockIdx.y*width + threadIdx.x + window_size]
			- iiPyramid[blockIdx.y*width + threadIdx.x + window_size]
			+ iiPyramid[blockIdx.y*width + threadIdx.x + window_size + window_size*width];
	norm_factor /= (window_size);
	norm_factor = norm_factor - mean*mean;
	norm_factor = (norm_factor > 0) * sqrtf(norm_factor) + (norm_factor <= 0);
	for (int i = 0; i < num_stages; i++) {
		pass += 1*stages[i].getValue<unsigned int>(iiPyramid + offset, norm_factor, width);
	}
	activationMask[blockIdx.y*width + threadIdx.x] = (unsigned char) (255*((1.0*pass)/num_stages));
}

void testGpuViolaJones(Image *faces, int numImgs, bool display) {

	cout << "Using GPU..." << endl << endl;

	// True positive and false positive variables
	int *tp = new int, *fp = new int; 
	*tp = *fp = 0;

	MyImage imageObj;
	MyImage *image = &imageObj;
	clock_t start = clock();
	for (int n = 0; n < numImgs; n++) {
		if (PRINT)
			cout << "Image " << faces[n].im_name << endl;

		// Ground truth bbox
		Rect gt = Rect(faces[n].x, faces[n].y, faces[n].w, faces[n].h);

		image->data = faces[n].grayscale.data;
		image->width = faces[n].grayscale.cols;
		image->height = faces[n].grayscale.rows;
		image->maxgrey = 255;

		std::vector<MyRect> result;
		detect_faces(image->width, image->height, result, image, SCALING, MIN_NEIGH);

		// No faces detected
		if (result.size() == 0) {
			if (PRINT)
				cout << "None detected :(" << endl << endl;
			continue;
		}

		//cout << "Detected " << result.size() << " images" << endl;

		// Calc IOU with first face
		MyRect last = result.back();
		Rect pred = Rect(last.x, last.y, last.width, last.height);
		IOU(pred, gt, tp, fp);

		if (display) {
			// Draw predicted faces
			rectangle(faces[n].image, pred, Scalar(255, 0, 0), 2); // GREEN

			// Draw gt face
			rectangle(faces[n].image, gt, Scalar(0, 0, 255), 2);  // RED

			imshow("GPU Result", faces[n].image);
			waitKey(1);
		}
	}

	printf("Final GPU accuracy: %d/%d = %f\n", *tp, numImgs, (float)*tp / numImgs);
	printf("Final GPU false positives: %d/%d = %f\n", *fp, numImgs, (float)*fp / numImgs);
	printf("Time elapsed: %.8lfs\n\n", (clock() - start) / (double)CLOCKS_PER_SEC);
}

/*
*	Only GPU face detection
*/
void gpuWebcam() {

	cout << "Starting webcam..." << endl << endl;

	// capture from web camera init
	VideoCapture cap(0);
	//cap.set(CV_CAP_PROP_FRAME_WIDTH, 720);
	//cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

	cout << "Camera found!" << endl;
	if (!cap.isOpened()) {
		cout << "Error opening video stream or file" << endl;
		return;
	}

	Mat img;
	MyImage imageObj;
	MyImage *image = &imageObj;
	for (;;) {

		// Image from camera to Mat
		cap.read(img);

		// Convert image to grayscale
		Mat gray_face;
		cvtColor(img, gray_face, CV_BGR2GRAY);
		
		image->data = gray_face.data;
		image->width = gray_face.cols;
		image->height = gray_face.rows;
		image->maxgrey = 255;

		std::vector<MyRect> result;
		detect_faces(image->width, image->height, result, image, SCALING, MIN_NEIGH);

		if (result.size() > 0) {
			for (int i = 0; i < result.size(); i++) {
				Rect pred = Rect(result[i].x, result[i].y, result[i].width, result[i].height);
				rectangle(img, pred, Scalar(255, 0, 0), 2); // GREEN
			}
		}

		imshow("GPU", img);
		char c = waitKey(10);
		if (c == 'b') {
			break; //break when b is pressed
		}

	}

}

/*
*	Displays ALL faces after detection
*/
void webcamGeneral() {

	cout << "Starting webcam..." << endl << endl;
	cout << "GPU is red. CPU is blue." << endl << endl;

	// capture from web camera init
	VideoCapture cap(0);
	//cap.set(CV_CAP_PROP_FRAME_WIDTH, 720);
	//cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

	cout << "Camera found!" << endl;
	if (!cap.isOpened()) {
		cout << "Error opening video stream or file" << endl;
		return;
	}

	CascadeClassifier face_cascade;
	face_cascade.load(CASCADE_PATH);

	Mat img;
	MyImage imageObj;
	MyImage *image = &imageObj;
	for (;;) {

		// Image from camera to Mat
		cap.read(img);

		// GPU CODE
		Mat gray_face;
		cvtColor(img, gray_face, CV_BGR2GRAY);

		image->data = gray_face.data;
		image->width = gray_face.cols;
		image->height = gray_face.rows;
		image->maxgrey = 255;

		std::vector<MyRect> result;
		detect_faces(image->width, image->height, result, image, SCALING, MIN_NEIGH);

		if (result.size() > 0) {
			for (int i = 0; i < result.size(); i++) {
				Rect pred = Rect(result[i].x, result[i].y, result[i].width, result[i].height);
				rectangle(img, pred, Scalar(0, 0, 255), 2); // RED
			}
		}

		// CPU CODE
		// Container of faces
		vector<Rect> faces;

		// Detect faces
		face_cascade.detectMultiScale(img, faces, SCALING, MIN_NEIGH, 0 | CV_HAAR_SCALE_IMAGE, Size(WIN_SIZE, WIN_SIZE));

		// To draw rectangles around detected faces
		if (faces.size() > 0) {
			for (unsigned i = 0; i<faces.size(); i++)
				rectangle(img, faces[i], Scalar(255, 0, 0), 2); // BLUE
		}


		imshow("CPU and GPU", img);
		char c = waitKey(10);
		if (c == 'b') {
			break; //break when b is pressed
		}

	}

}


/*
*	Only displays the LAST face after detection
*/
void webcamTest() {

	cout << "Starting webcam..." << endl << endl;
	cout << "GPU is red. CPU is blue." << endl << endl;

	// capture from web camera init
	VideoCapture cap(0);
	//cap.set(CV_CAP_PROP_FRAME_WIDTH, 720);
	//cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

	cout << "Camera found!" << endl;
	if (!cap.isOpened()) {
		cout << "Error opening video stream or file" << endl;
		return;
	}

	CascadeClassifier face_cascade;
	face_cascade.load(CASCADE_PATH);

	Mat img;
	MyImage imageObj;
	MyImage *image = &imageObj;
	for (;;) {

		// Image from camera to Mat
		cap.read(img);

		// GPU CODE
		Mat gray_face;
		cvtColor(img, gray_face, CV_BGR2GRAY);

		image->data = gray_face.data;
		image->width = gray_face.cols;
		image->height = gray_face.rows;
		image->maxgrey = 255;

		std::vector<MyRect> result;
		detect_faces(image->width, image->height, result, image, SCALING, MIN_NEIGH);

		if (result.size() > 0) {
			MyRect last = result.back();
			Rect pred = Rect(last.x, last.y, last.width, last.height);
			rectangle(img, pred, Scalar(0, 0, 255), 2); // RED
		}

		// CPU CODE
		// Container of faces
		vector<Rect> faces;

		// Detect faces
		face_cascade.detectMultiScale(img, faces, SCALING, MIN_NEIGH, 0 | CV_HAAR_SCALE_IMAGE, Size(WIN_SIZE, WIN_SIZE));

		if (faces.size() > 0) {
		// To draw rectangles around detected faces
			Rect face = faces.back();
			rectangle(img, face, Scalar(255, 0, 0), 2); // BLUE
		}

		imshow("CPU and GPU", img);
		char c = waitKey(10);
		if (c == 'b') {
			break; //break when b is pressed
		}

	}

}