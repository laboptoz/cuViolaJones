#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>
#include "load_images.hpp"
#include "paths.hpp"
#include "macros.hpp"

using namespace cv;
using namespace std;

//A Haar Cascade Classifier implementation using OpenCV for reference
void cpuViolaJones(Mat face, String cascade_path) {

	//Load the cascade classifier
	CascadeClassifier face_cascade; 
	if (!face_cascade.load(cascade_path)) {
		printf("Error loading face cascade");
	}

	//Convert the face to gray
	Mat gray_face;
	cvtColor(face, gray_face, CV_BGR2GRAY);
	equalizeHist(gray_face, gray_face);

	//Detect faces
	vector<Rect> faces;
	face_cascade.detectMultiScale(gray_face, faces, 1.1, 3, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

	//Draw faces
	for (int i = 0; i < faces.size(); i++) {
		rectangle(face, faces[i], Scalar(255, 0, 0),3);
	}

	//Show image
	imshow("CPU Result", face);
}

void cpuVJ(Image img, String cascade_path, int *tp, int *fp, bool display) {
	
	// Ground truth bbox
	Rect gt = Rect(img.x, img.y, img.w, img.h);
	if (PRINT)
		cout << "Image " << img.im_name << endl;

	//Load the cascade classifier
	CascadeClassifier face_cascade;
	if (!face_cascade.load(cascade_path)) {
		printf("Error loading face cascade");
	}

	//Convert the face to gray
	Mat gray_face;
	cvtColor(img.image, gray_face, CV_BGR2GRAY);
	equalizeHist(gray_face, gray_face);

	//Detect faces
	vector<Rect> faces;
	face_cascade.detectMultiScale(gray_face, faces, SCALING, MIN_NEIGH, 0 | CV_HAAR_SCALE_IMAGE, Size(WIN_SIZE, WIN_SIZE));

	// No faces detected
	if (faces.size() == 0) {
		if (PRINT)
			cout << "None detected :(" << endl << endl;
		return;
	}

	// Calc IOU with first face
	Rect face = faces.back();
	IOU(face, gt, tp, fp);

	if (display) {
		//Draw predicted faces
		rectangle(img.image, face, Scalar(255, 0, 0), 2); // BLUE
		printf("Predicted bbox (Blue): %d, %d, %d, %d\n", face.x, face.y, face.width, face.height);

		// Draw gt face
		rectangle(img.image, gt, Scalar(0, 0, 255), 2);  // RED
		printf("GT bbox (Red): %d, %d, %d, %d\n\n", img.x, img.y, img.w, img.h);

		//Show image
		imshow("CPU Result", img.image);
		waitKey(1);
	}
}

void testCpuViolaJones(Image * imgs, int numImgs, bool display) {

	cout << "Using CPU..." << endl << endl;
	// Limit images
	//numImgs = 100;

	int *tp = new int, *fp = new int; // true positive and false positive
	*tp = *fp = 0;

	clock_t start = clock();
	for (int i = 0; i < numImgs; i++) {
		cpuVJ(imgs[i], CASCADE_PATH, tp, fp, display);
	}

	printf("Final CPU accuracy: %d/%d = %f\n", *tp, numImgs, (float)*tp/numImgs);
	printf("Final CPU false positives: %d/%d = %f\n", *fp, numImgs, (float)*fp / numImgs);
	printf("Time elapsed: %.8lfs\n\n", (clock() - start) / (double)CLOCKS_PER_SEC);
}


/*
*	Only CPU face detection
*/
void cpuWebcam() {

	cout << "Starting webcam..." << endl << endl;

	// capture from web camera init
	VideoCapture cap(0);
	//cap.set(CV_CAP_PROP_FRAME_WIDTH, 720);
	//cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

	CascadeClassifier face_cascade;
	face_cascade.load(CASCADE_PATH);

	cout << "Camera found!" << endl;
	if (!cap.isOpened()) {
		cout << "Error opening video stream or file" << endl;
		return;
	}

	/*
	while (1) {
		cout << "Running.." << endl;
		//Mat img;
		//cap >> img;
		//imshow("Window 1", img);

		Mat img2;
		bool success = cap.read(img2);
		imshow(window_name, img2);

		if (success == false) {
			cout << "Video camera is disconnected" << endl;
			break;
		}

		char c = waitKey(10);
		if (c == 'b') {
			break; //break when b is pressed
		}
	}
	*/
	Mat img;
	for (;;){

		// Image from camera to Mat
		cap.read(img);

		// Container of faces
		vector<Rect> faces;

		// Detect faces
		face_cascade.detectMultiScale(img, faces, SCALING, MIN_NEIGH, 0 | CV_HAAR_SCALE_IMAGE, Size(WIN_SIZE, WIN_SIZE));

		// To draw rectangles around detected faces
		if (faces.size() > 0) {
			for (unsigned i = 0; i<faces.size(); i++)
				rectangle(img, faces[i], Scalar(255, 0, 0), 2); // BLUE
		}
		
		imshow("CPU", img);
		char c = waitKey(10);
		if (c == 'b') {
			break; //break when b is pressed
		}

	}
	
}
