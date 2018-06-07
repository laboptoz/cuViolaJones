#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include "Filter.cuh"
#include "paths.hpp"
#include "cuda_error_check.h"

Stage * loadParametersToGPU() {
	FILE * stage_info_file = fopen(INFO_PATH, "r");
	FILE * filter_info_file = fopen(CLASS_PATH, "r");
	unsigned int num_stages;
	fscanf(stage_info_file, "%u\n", &num_stages);
	printf("There are %u stages.\n", num_stages);
	
	//Allocate GPU Stage space
	Stage * stages_gpu;
	CHECK(cudaMalloc(&stages_gpu, sizeof(Stage)*num_stages));

	Stage * stages_cpu = new Stage[num_stages];
	for (int i = 0; i < num_stages; i++) {
		unsigned int num_filters;
		fscanf(stage_info_file, "%u\n", &num_filters);
		stages_cpu[i].num_filters = num_filters;
		stages_cpu[i].filters = (Filter *)malloc(sizeof(Filter)*num_filters);
		for (int j = 0; j < stages_cpu[i].num_filters; j++) {
			unsigned int x1, y1, width1, height1;
			unsigned int x2, y2, width2, height2;
			unsigned int x3, y3, width3, height3;
			int weight1, weight2, weight3;
			int filt_threshold, alpha1, alpha2;
			fscanf(filter_info_file, "%u\n%u\n%u\n%u\n%d\n", &x1, &y1, &width1, &height1, &weight1);
			fscanf(filter_info_file, "%u\n%u\n%u\n%u\n%d\n", &x2, &y2, &width2, &height2, &weight2);
			fscanf(filter_info_file, "%u\n%u\n%u\n%u\n%d\n", &x3, &y3, &width3, &height3, &weight3);
			fscanf(filter_info_file, "%d\n%d\n%d\n", &filt_threshold, &alpha1, &alpha2);
			stages_cpu[i].filters[j] = Filter(x1, y1, width1, height1, weight1,
												x2, y2, width2, height2, weight2,
												x3, y3, width3, height3, weight3,
												filt_threshold, alpha1, alpha2);
		}

		int threshold;
		fscanf(filter_info_file, "%d\n", &threshold);
		stages_cpu[i].threshold = threshold;
	}

	Stage * stages_to_copy = (Stage *)malloc(sizeof(Stage)*num_stages);
	memcpy(stages_to_copy, stages_cpu, sizeof(Stage)*num_stages);
	for (int i = 0; i < num_stages; i++) {
		CHECK(cudaMalloc(&(stages_to_copy[i].filters), sizeof(Filter)*stages_to_copy[i].num_filters));
	}

	CHECK(cudaMemcpy(stages_gpu, stages_to_copy, sizeof(Stage)*num_stages, cudaMemcpyHostToDevice));

	for (int i = 0; i < num_stages; i++) {
		CHECK(cudaMemcpy(stages_to_copy[i].filters, stages_cpu[i].filters, sizeof(Filter)*stages_cpu[i].num_filters, cudaMemcpyHostToDevice));
	}

	free(stages_to_copy);
	free(stages_cpu);

	fclose(stage_info_file);
	fclose(filter_info_file);
	return stages_gpu;
}