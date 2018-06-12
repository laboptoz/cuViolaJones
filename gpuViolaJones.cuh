#pragma once
#include <stdio.h>
#include <cuda_runtime.h>
//#include "cuNNII.h"
#include "Filter.cuh"
#include "haar.cuh"
#include "image.h"
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

	clock_t start = clock();
	for (int n = 0; n < numImgs; n++) {
		if (PRINT)
			cout << "Image " << faces[n].im_name << endl;

		// Ground truth bbox
		Rect gt = Rect(faces[n].x, faces[n].y, faces[n].w, faces[n].h);

		MyImage imageObj;
		MyImage *image = &imageObj;
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
			rectangle(faces[n].image, pred, Scalar(255, 0, 0), 2); // BLUE

			// Draw gt face
			rectangle(faces[n].image, gt, Scalar(0, 0, 255), 2);  // RED

			imshow("GPU Result", faces[n].image);
			//ostringstream name;
			//name << "bush_" << n << ".jpg";
			//imwrite(name.str().c_str(), faces[n].image);
			waitKey(1);
		}
	}

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

	MyImage imageObj;
	MyImage *image = &imageObj;
	image->data = gray_face.data;
	image->width = gray_face.cols;
	image->height = gray_face.rows;
	image->maxgrey = 255;

	std::vector<MyRect> result;
	detect_faces(image->width, image->height, result, image, SCALING, MIN_NEIGH);

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