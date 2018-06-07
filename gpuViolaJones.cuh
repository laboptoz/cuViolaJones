#include <cuda_runtime.h>
//#include "cuIntegralImage.cuh"
#include "cuNNII.h"
#include "Filter.cuh"
#include "cuda_error_check.h"
#define TEST 1

//TESTING FUNCTION
__global__ void testfunc(float * input, unsigned int width);

void gpuViolaJones() {
	unsigned char * input = new unsigned char[8*8];
	for (int i = 0; i < 8 * 8; i++) {
		input[i] = i + 1;
	}
	unsigned int * pyramidSizes = nullptr;
	float ** gpuPyramid = generateImagePyramid(input, &pyramidSizes, 2, 8, 8, 1.2);
#if TEST
	printf("Testing filter.\n");
	testfunc <<<1,1>>>(gpuPyramid[0], pyramidSizes[0]);
#endif
}


__global__ void testfunc(float * input, unsigned int width) {
	Filter f = Filter(0, 0, 2, 2, -1, 0, 2, 2, 2, 1, 2, 0, 3);
	float num =  f.getValue(input, width);
	printf("\n\n%f\n\n", num);
}