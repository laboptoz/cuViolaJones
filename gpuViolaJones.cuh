#include <cuda_runtime.h>
#include "cuNNII.h"
#include "Filter.cuh"
#include "cuda_error_check.h"
#include "parameter_loader.h"
#include "Point.h"

std::vector<vj::Point> gpuViolaJones(unsigned char * img, unsigned int width, unsigned int height, unsigned int min_size, float scale) {
	
	unsigned int * num_stages = new unsigned int;
	Stage * stages_gpu = loadParametersToGPU(num_stages);


	unsigned int * iiPyramidSizes = nullptr;
	unsigned int iiPyramidDepth = 0;
	float ** iiPyramid_gpu = generateImagePyramid<false>(img, &iiPyramidSizes, &iiPyramidDepth, min_size, width, height, scale);

	unsigned int * viiPyramidSizes = nullptr;
	unsigned int viiPyramidDepth = 0;
	float ** viiPyramid_gpu = generateImagePyramid<true>(img, &viiPyramidSizes, &viiPyramidDepth, min_size, width, height, scale);

	std::vector<vj::Point> faces;
	return faces;
}