#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "cpuViolaJones.hpp"
#include "paths.hpp"
#include "gpuViolaJones.cuh"
#include "load_images.hpp"

using namespace std;
using namespace cv;


int main(int argc, char** argv )
{
    Mat image;
    image = imread(FACE_PATH, 1 );
	String face_cascade_path = CASCADE_PATH;
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
	
	Mat gray_face;
	cvtColor(image, gray_face, CV_BGR2GRAY);
	//imshow("gray", gray_face);
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
	
	// Load test images
	int *numImgs = new int;
	Image *imgs = loadData(LABEL_PATH, IMAGE_PATH, numImgs);
	imshow("", imgs[1699].image);
	waitKey(0);

	printf("%d, %d\n", height, width);
	gpuViolaJones();
	//cpuViolaJones(image, face_cascade_path);
    waitKey(0);

    return 0;
}
