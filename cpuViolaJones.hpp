#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>
#include "load_images.hpp"

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

int cpuVJ(Image img, String cascade_path) {
	
	// Ground truth bbox
	Rect gt = Rect(img.x, img.y, img.w, img.h);
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
	face_cascade.detectMultiScale(gray_face, faces, 1.1, 3, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

	// No faces detected
	if (faces.size() == 0) 
		return 0;

	// Calc IOU with first face
	Rect face = faces.front();
	int detected = IOU(face, gt);

	//Draw predicted faces
	rectangle(img.image, face, Scalar(255, 0, 0), 2); // BLUE
	printf("Predicted bbox (Blue): %d, %d, %d, %d\n", face.x, face.y, face.width, face.height);

	// Draw gt face
	rectangle(img.image, gt, Scalar(0, 0, 255), 2);  // RED
	printf("GT bbox (Red): %d, %d, %d, %d\n\n", img.x, img.y, img.w, img.h);

	//Show image
	imshow("CPU Result", img.image);
	waitKey(1);

	return detected;
}

void testCpuViolaJones(Image * imgs, int numImgs, String face_cascade_path) {
	//numImgs = 1;
	int detected = 0;
	for (int i = 0; i < numImgs; i++) {
		detected += cpuVJ(imgs[i], face_cascade_path);
	}
	printf("Final CPU accuracy = %d/%d = %f\n", detected, numImgs, (float)detected/numImgs);
}