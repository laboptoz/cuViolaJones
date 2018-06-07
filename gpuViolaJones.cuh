#include <cuda_runtime.h>
#include "cuNNII.h"
#include "Filter.cuh"
#include "cuda_error_check.h"
#include "parameter_loader.h"
#define TEST 0

void gpuViolaJones() {
	
	//PLACEHOLDER VARIABLES UNTIL WE GET THE INPUTS TO THIS FUNCTION SET UP
	unsigned int width = 8;
	unsigned int height = 8;
	unsigned char * input = new unsigned char[width * height];
	for (int i = 0; i < width * height; i++) {
		input[i] = i + 1;
	}
	unsigned int min_size = 2;
	float scale = 1.2;

	//END PLACEHOLDERS


	Stage * stages_gpu = loadParametersToGPU();

	unsigned int * iiPyramidSizes = nullptr;
	unsigned int iiPyramidDepth = 0;
	float ** iiPyramid_gpu = generateImagePyramid<false>(input, &iiPyramidSizes, &iiPyramidDepth, min_size, width, height, scale);

	unsigned int * viiPyramidSizes = nullptr;
	unsigned int viiPyramidDepth = 0;
	float ** viiPyramid_gpu = generateImagePyramid<true>(input, &viiPyramidSizes, &viiPyramidDepth, min_size, width, height, scale);



#if TEST
	unsigned char * input = new unsigned char[8*8];
	for (int i = 0; i < 8 * 8; i++) {
		input[i] = i + 1;
	}
	unsigned int * pyramidSizes = nullptr;
	unsigned int pyramidDepth = 0;
	float ** gpuPyramid = generateImagePyramid<false>(input, &pyramidSizes, &pyramidDepth, 2, 8, 8, 1.2);
	printf("Pyramid depth is %u\n", pyramidDepth);

#endif
}