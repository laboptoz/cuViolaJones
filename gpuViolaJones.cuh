#pragma once
#include <stdio.h>
#include <cuda_runtime.h>
#include "Filter.cuh"
#include "haar.cuh"
#include "cuda_error_check.h"
#include "parameter_loader.h"
#include <sstream>

using namespace std;
using namespace cv;

/*
*	Main test for GPU code
*/
void testGpuViolaJones(Image *faces, int numImgs, bool display) {

	cout << "Using GPU..." << endl << endl;

	// True positive and false positive variables
	int *tp = new int, *fp = new int; 
	*tp = *fp = 0;
	
	//Start timing
	clock_t start = clock();
	
	//Load stage information from file
	unsigned int * num_stages = new unsigned int;
	Stage * stages_gpu = loadParametersToGPU(num_stages);
	
	//Loop over number of images
	for (int n = 0; n < numImgs; n++) {
		if (PRINT)
			cout << "Image " << faces[n].im_name << endl;

		// Ground truth bbox
		Rect gt = Rect(faces[n].x, faces[n].y, faces[n].w, faces[n].h);

		//Set image data
		ImageUnion *image = new ImageUnion();
		image->dataChar = faces[n].grayscale.data;
		image->width = faces[n].grayscale.cols;
		image->height = faces[n].grayscale.rows;

		//The result of the detect faces function
		std::vector<Rectangle> result;
		
		//Call detect faces function
		detect_faces(image->width, image->height, result, image, SCALING, MIN_NEIGH, num_stages, stages_gpu);

		// No faces detected
		if (result.size() == 0) {
			if (PRINT)
				cout << "None detected :(" << endl << endl;
			continue;
		}

		//cout << "Detected " << result.size() << " images" << endl;

		// Calc IOU with first face
		Rectangle last = result.back();
		Rect pred = Rect(last.x, last.y, last.width, last.height);
		IOU(pred, gt, tp, fp);

		if (display) {
			// Draw predicted faces
			rectangle(faces[n].image, pred, Scalar(255, 0, 0), 2); // BLUE

			// Draw gt face
			rectangle(faces[n].image, gt, Scalar(0, 0, 255), 2);  // RED

			//Display the face
			imshow("GPU Result", faces[n].image);
			waitKey(1);
		}
	}

	//Print GPU test information
	printf("Final GPU accuracy: %d/%d = %f\n", *tp, numImgs, (float)*tp / numImgs);
	printf("Final GPU false positives: %d/%d = %f\n", *fp, numImgs, (float)*fp / numImgs);
	printf("Time elapsed: %.8lfs\n\n", (clock() - start) / (double)CLOCKS_PER_SEC);
}

/*
*	Detects only one face on GPU
*/
void gpuSingleDetection(Mat face) {

	Mat gray_face;
	cvtColor(face, gray_face, CV_BGR2GRAY);

	ImageUnion *image = new ImageUnion();
	image->dataChar = gray_face.data;
	image->width = gray_face.cols;
	image->height = gray_face.rows;

	unsigned int * num_stages = new unsigned int;
	Stage * stages_gpu = loadParametersToGPU(num_stages);

	std::vector<Rectangle> result;
	detect_faces(image->width, image->height, result, image, SCALING, MIN_NEIGH, num_stages, stages_gpu);

	if (result.size() > 0) {
		for (int i = 0; i < result.size(); i++) {
			Rect pred = Rect(result[i].x, result[i].y, result[i].width, result[i].height);
			rectangle(face, pred, Scalar(0, 255, 0), 2); // GREEN
		}
	}
	imshow("GPU Result", face);
	waitKey(0);

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
	ImageUnion *image = new ImageUnion();
	unsigned int * num_stages = new unsigned int;
	Stage * stages_gpu = loadParametersToGPU(num_stages);
	for (;;) {

		// Image from camera to Mat
		cap.read(img);

		// Convert image to grayscale
		Mat gray_face;
		cvtColor(img, gray_face, CV_BGR2GRAY);
		
		image->dataChar = gray_face.data;
		image->width = gray_face.cols;
		image->height = gray_face.rows;

		std::vector<Rectangle> result;
		detect_faces(image->width, image->height, result, image, SCALING, MIN_NEIGH, num_stages, stages_gpu);

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
*	Displays ALL faces after detection of CPU and GPU
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
	ImageUnion *image = new ImageUnion();
	unsigned int * num_stages = new unsigned int;
	Stage * stages_gpu = loadParametersToGPU(num_stages);
	for (;;) {

		// Image from camera to Mat
		cap.read(img);

		// GPU CODE
		Mat gray_face;
		cvtColor(img, gray_face, CV_BGR2GRAY);

		image->dataChar = gray_face.data;
		image->width = gray_face.cols;
		image->height = gray_face.rows;

		std::vector<Rectangle> result;
		detect_faces(image->width, image->height, result, image, SCALING, MIN_NEIGH, num_stages, stages_gpu);

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
	ImageUnion *image = new ImageUnion();
	unsigned int * num_stages = new unsigned int;
	Stage * stages_gpu = loadParametersToGPU(num_stages);
	for (;;) {

		// Image from camera to Mat
		cap.read(img);

		// GPU CODE
		Mat gray_face;
		cvtColor(img, gray_face, CV_BGR2GRAY);

		image->dataChar = gray_face.data;
		image->width = gray_face.cols;
		image->height = gray_face.rows;

		std::vector<Rectangle> result;
		detect_faces(image->width, image->height, result, image, SCALING, MIN_NEIGH, num_stages, stages_gpu);

		if (result.size() > 0) {
			Rectangle last = result.back();
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

		// detect memory usage
#if REPORT_GMEM == 1
		size_t remainMem, totalMem;
		cudaMemGetInfo(&remainMem, &totalMem);
		printf("\rGPU Remaining Mem: %d", (remainMem / (1024 * 1024)));
#endif

		imshow("CPU and GPU", img);
		char c = waitKey(10);
		if (c == 'b') {
			break; //break when b is pressed
		}

	}

}
