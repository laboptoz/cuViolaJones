#include <opencv2/opencv.hpp>

using namespace cv;

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
	std::vector<Rect> faces;
	face_cascade.detectMultiScale(gray_face, faces, 1.1, 3, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

	//Draw faces
	for (int i = 0; i < faces.size(); i++) {
		rectangle(face, faces[i], Scalar(255, 0, 0),3);
	}

	//Show image
	imshow("CPU Result", face);
}