#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>
#include "load_images.hpp"
#include "paths.hpp"

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
	if (faces.size() == 0) {
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

float ** cpuIntegralImage(unsigned char * original,
					 unsigned int ** sizes_ptr,
		 			 unsigned int * depth,
					 unsigned int min_size,
					 unsigned int width,
					 unsigned int height,
					 float scale) {
	//DETERMINE PYRAMID DEPTH
	unsigned int scaled_width = width;
	unsigned int scaled_height = height;
	*depth = 0;
	unsigned int sum_size = 0;

	if (min_size <= 3) {
		min_size = 4;
	}
	while (scaled_width >= min_size && scaled_height >= min_size) {
		(*depth)++;
		sum_size += scaled_width*scaled_height;
		scaled_width = round(scaled_width / scale);
		scaled_height = round(scaled_height / scale);
	}
	printf("Image pyramid depth is: %u\n", *depth);

	// Malloc the images
	float * imagePyramid = new float[sum_size * sizeof(float)];

	float ** integralimages = new float *[*depth];

	//Assign pyramid sizes to argument 'sizes'
	sum_size = 0;
	scaled_width = width;
	scaled_height = height;
	*sizes_ptr = (unsigned int *)malloc(2 * (*depth) * sizeof(unsigned int)); // 2 b/c width and height
	unsigned int * sizes = *sizes_ptr;
	for (int i = 0; i < *depth; i++) {
		integralimages[i] = imagePyramid + sum_size;
		sum_size += scaled_width*scaled_height;
		sizes[2 * i] = scaled_width;
		sizes[2 * i + 1] = scaled_height;
		scaled_width = round(scaled_width / scale);
		scaled_height = round(scaled_height / scale);
	}

	// Convert orignal image to a cv::Mat
	Mat img = Mat(width, height, CV_8UC1, original);
	cout << "Original: " << endl;
	cout << img << endl << endl;

	// First integral image
	Mat integ, tmp;
	tmp = img;
	integral(img, tmp);
	cout << "Pyramid level 0:" << endl;
	cout << tmp << endl << endl;
	//unsigned char * data = integ.data;


	scaled_width = width;
	scaled_height = height;
	// Generate remaining integral images
	for (int i = 0; i < *depth; i++) {
		printf("Pyramid level: %u\n", i);
		cout << scaled_width << " "<< scaled_height << endl;
		// Downsample
		Mat tmp1;
		resize(img, tmp1, Size(scaled_width, scaled_height), 0, 0, CV_INTER_LINEAR);
		// Integral image
		Mat tmp2;
		integral(tmp1, tmp2);
		cout << tmp2 << endl;
		Mat img = tmp2.clone();
		// Change size
		scaled_width = round(scaled_width / scale);
		scaled_height = round(scaled_height / scale);
		// Save data
		//float *idx = integralimages[i];
		//memcpy(imagePyramid[idx], tmp2.data, scaled_width*scaled_height);
	}

	return integralimages;
}

void testCpuII() {
	unsigned int width = 8;
	unsigned int height = 8;
	unsigned char * img = new unsigned char[width * height];
	for (int i = 0; i < width * height; i++) {
		img[i] = 1;
	}
	unsigned int win_size = 2;
	float scale = 1.2;

	unsigned int * iiPyramidSizes = nullptr;
	unsigned int iiPyramidDepth = 0;
	float ** integralImages = cpuIntegralImage(img,
								 &iiPyramidSizes, 
								 &iiPyramidDepth, 
								 win_size, 
								 width, 
								 height, 
								 scale);
}

/*
void cpuWebcam(String face_cascade_path) {

	cout << "Starting webcam..." << endl << endl;

	// capture from web camera init
	VideoCapture cap(0);
	cout << "Camera found!" << endl;

	if (!cap.open(0)) {
		cout << "Webcam not open" << endl;
		return;
	}
	for (;;) {
		cout << "Starting loop" << endl;
		Mat frame;
		cap >> frame;
		if (frame.empty()) {
			cout << "Frame is empty" << endl;
			break; // end of video stream
		}
		cout << "Displaying image" << endl;
		imshow("this is you, smile! :)", frame);
		if (waitKey(10) == 27) {
			break; // stop capturing by pressing ESC 
		}
	}

	
	Mat img;

	// Initialize the inbuilt Harr Cascade frontal face detection
	// Below mention the path of where your haarcascade_frontalface_alt2.xml file is located
	CascadeClassifier face_cascade;
	if (!face_cascade.load(face_cascade_path)) {
		printf("Error loading face cascade");
	}

	for (;;) {

		// Image from camera to Mat
		cap >> img;

		// obtain input image from source
		cap.retrieve(img);

		// Just resize input image if you want
		resize(img, img, Size(1000, 640));

		// Container of faces
		vector<Rect> faces;

		// Detect faces
		face_cascade.detectMultiScale(img, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(140, 140));

		//Show the results
		// Draw circles on the detected faces
		for (int i = 0; i < faces.size(); i++) {
			Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
			ellipse(img, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
		}

		// To draw rectangles around detected faces
		for (unsigned i = 0; i<faces.size(); i++)
			rectangle(img,faces[i], Scalar(255, 0, 0), 2, 1);

		imshow("CPU", img);
		int key2 = waitKey(20);

	}
}
*/