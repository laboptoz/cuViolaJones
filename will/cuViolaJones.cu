#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "cpuViolaJones.hpp"
#include "paths.hpp"
//#include "gpuViolaJones.cuh"
#include "haar.cuh"

#include <stdlib.h>
#include "image.h"
//#include "stdio-wrapper.h"


using namespace cv;
using namespace std;


void run_vj_gpu(Mat gray_face) {

	int mode = 1;
	int i;

	/* detection parameters */
	float scaleFactor = 1.2;
	int minNeighbours = 1;

	MyImage imageObj;
	MyImage *image = &imageObj;
	image->data = gray_face.data;
	image->width = gray_face.cols;
	image->height = gray_face.rows;
	image->maxgrey = 255;
	image->flag = 1;

	//printf("-- loading cascade classifier --\r\n");

	//myCascade cascadeObj;
	//myCascade *cascade = &cascadeObj;
	//MySize minSize = { 20, 20 };
	//MySize maxSize = { 0, 0 };

	///* classifier properties */
	//cascade->n_stages = 25;
	//cascade->total_nodes = 2913;
	//cascade->orig_window_size.height = 24;
	//cascade->orig_window_size.width = 24;

	//printf("-- load filters and weights --\n");
	//readTextClassifier();

	//printf("-- detecting faces --\r\n");
	std::vector<MyRect> result;
	detect_faces(image->width, image->height, result, image);

	for (i = 0; i < result.size(); i++){
		MyRect r = result[i];
		drawRectangle(image, r);
	}

	imshow("CPU Result", gray_face);

	///* delete image and free classifier */
	//releaseTextClassifier();
	////freeImage(image);
}

void crop_image(Mat image) {
	cv::Rect rec(100, 165, 40, 40);
	imwrite(CROP_OUT, image(rec));
	//imshow("", image(rec));
}


int main(int argc, char** argv )
{
    Mat image;
    image = imread(FACE_PATH_2, 1 );
	String face_cascade_path = CASCADE_PATH;
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
	
	Mat gray_face;
	cvtColor(image, gray_face, CV_BGR2GRAY);
	unsigned char * face;
	unsigned int height = 0;
	unsigned int width = 0;
	if (gray_face.isContinuous()) {
		face = gray_face.data;
		height = gray_face.rows;
		width = gray_face.cols;
	}
	else {
		fprintf(stderr, "Stop\n");
	}

	printf("Image shape: %d * %d\n", height, width);

	//crop_image(gray_face);

	//imshow("", gray_face);
	run_vj_gpu(gray_face);

	//sliding_window();

    waitKey(0);

    return 0;
}