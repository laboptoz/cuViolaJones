#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "cpuViolaJones.hpp"

using namespace cv;


int main(int argc, char** argv )
{
    Mat image;
    image = imread("C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/faces.jpg", 1 );
	Mat smallimage;
	resize(image, smallimage, Size(), 0.75, 0.75);
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }

	cpuViolaJones(image);
    waitKey(0);

    return 0;
}