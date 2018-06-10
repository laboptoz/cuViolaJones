#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>
#include <stdio.h>
#include <string.h>
#include <iostream> 
#include <fstream>
#include "macros.hpp"

#define THRESHOLD .5

using namespace std;
using namespace cv;

struct Image {
	Mat image;
	Mat grayscale;
	int x;	
	int y;
	int w;
	int h;
	string im_name;
};


/* 
*	Reads the text file that has the image names and bbox 
*	Returns an array of images and the bboxes
*/
Image * loadData(string textFile, string imagePath, int *numImgs) {
	
	cout << "Loading images..." << endl << endl;

	ifstream labels(textFile);

	// new lines will be skipped unless we stop it from happening:    
	labels.unsetf(ios_base::skipws);

	// count the newlines with an algorithm specialized for counting:
	int line_count = count(
		istream_iterator<char>(labels),
		istream_iterator<char>(),
		'\n');

	//cout << "Number of images: " << line_count << "\n" << endl;

	// Store total number of images
	*numImgs = line_count; 

	// Return to beginning of file
	labels.clear();
	labels.seekg(0, ios::beg);

	Image *imgs = new Image[line_count];
	string line;
	int idx = 0;
	while (getline(labels, line)) {
		istringstream iss(line);
		vector<string> results(istream_iterator<string>{iss}, istream_iterator<string>()); // splits line with space delimiter
		Mat face = imread(imagePath + results[0], 1);		 // read and store image
		imgs[idx].image = face;
		cvtColor(face, imgs[idx].grayscale, CV_BGR2GRAY);	 // Convert to grayscale
		imgs[idx].im_name = results[0];						 // Store image name
		imgs[idx].x = stoi(results[1]);						 // stoi() converts string to integer
		imgs[idx].y = stoi(results[2]);
		imgs[idx].w = stoi(results[3]);
		imgs[idx].h = stoi(results[4]);
		idx++;
	}
	cout << "Loaded " << line_count << " images. \n" << endl;
	return imgs;
}


/*
*	Calculates the intersection over union of two bboxes
*	Returns if face detected or not
*	TODO: Fix arguments to correct data type
*/
void IOU(Rect pred, Rect gt, int *tp, int *fp) {
	
	float iou = 0.0;

	// calc intersection
	Rect r1 = pred & gt;
	float intersection = r1.width * r1.height;

	if (intersection > 0.0) { 
		// calc overlap
		float overlap = (pred.width*pred.height) + (gt.width*gt.height) - intersection;
		iou = intersection / overlap;
	}
	else {
		// false positive (detected but wrong face)
		if (PRINT)
			cout << "False positive" << endl << endl;
		*fp = *fp + 1;
		return;
	}

	if (PRINT)
		cout << "IOU: " << iou << endl << endl;;

	if (iou > THRESHOLD)
		*tp = *tp + 1;
}

void testIOU() {
	Rect pred = Rect(143,230,110,110);
	Rect gt = Rect(153,246,90,101);
	int *tp = new int; // true positive
	*tp = 0;
	int *fp = new int; // false positive
	*fp = 0;

	IOU(pred, gt, tp, fp);
	if(*tp)
		printf("Face detected! :D\n");
	else
		printf("Cannot find face! D:\n");
}