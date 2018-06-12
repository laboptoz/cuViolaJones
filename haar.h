#ifndef __HAAR_H__
#define __HAAR_H__

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define MAXLABELS 50

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	int width;
	int height;
	union {
		unsigned char* dataChar;
		int* dataInt;
	};
} ImageUnion;

typedef struct
{
	int width;
	int height;
} ImageDim;

typedef struct
{
	int x;
	int y;
	int width;
	int height;
} Rectangle;

//void groupRectangles(MyRect* _vec, int groupThreshold, float eps);
void groupRectangles(std::vector<Rectangle>& _vec, int groupThreshold, float eps);

/* draw white bounding boxes around detected faces */
void drawRectangle(ImageUnion* image, Rectangle r);

void detect_faces(unsigned int img_width, unsigned int img_height, std::vector<Rectangle> &allCandidates,
	ImageUnion* _img);
		
#ifdef __cplusplus
}

#endif

#endif
