#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "cpuViolaJones.hpp"
#include "paths.hpp"
//#include "gpuViolaJones.cuh"

using namespace cv;


int main(int argc, char** argv )
{
	FILE * test = fopen("text.txt", "w");
    Mat image;
    image = imread(FACE_PATH, 1 );
	String face_cascade_path = CASCADE_PATH;
	Mat smallimage;
	resize(image, smallimage, Size(), 0.75, 0.75);
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }

	cpuViolaJones(smallimage, face_cascade_path);
    waitKey(0);

    return 0;
}