#include <cuda_runtime.h>
#include "cuNNII.h"
#include "Filter.cuh"
#include "cuda_error_check.h"
#include "parameter_loader.h"
#define TEST 1

void gpuViolaJones() {
	Stage * stages_gpu = loadParametersToGPU();

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