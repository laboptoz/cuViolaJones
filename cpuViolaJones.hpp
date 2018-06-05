#include <opencv2/opencv.hpp>

using namespace cv;
void cpuViolaJones(Mat face) {
	String face_cascade_path = "C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/lib/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml";
	CascadeClassifier face_cascade; 
	if (!face_cascade.load(face_cascade_path)) {
		printf("Error loading face cascade");
	}
	Mat gray_face;
	cvtColor(face, gray_face, CV_BGR2GRAY);
	equalizeHist(gray_face, gray_face);
	std::vector<Rect> faces;
	face_cascade.detectMultiScale(gray_face, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

	for (int i = 0; i < faces.size(); i++) {
		rectangle(face, faces[i], Scalar(255, 0, 0),3);
	}

	imshow("Face result", face);
}