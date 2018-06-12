/*
 *  TU Eindhoven
 *  Eindhoven, The Netherlands
 *
 *  Name            :   haar.h
 *
 *  Author          :   Francesco Comaschi (f.comaschi@tue.nl)
 *
 *  Date            :   November 12, 2012
 *
 *  Function        :   Haar features evaluation for face detection
 *
 *  History         :
 *      12-11-12    :   Initial version.
 *
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program;  If not, see <http://www.gnu.org/licenses/>
 *
 * In other words, you are welcome to use, share and improve this program.
 * You are forbidden to forbid anyone else to use, share and improve
 * what you give them.   Happy coding!
 */

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

/* group all rectangles */
void groupRectangles(std::vector<Rectangle>& _vec, int groupThreshold, float eps);

/* draw white bounding boxes around detected faces */
void drawRectangle(ImageUnion* image, Rectangle r);

void detect_faces(unsigned int img_width, unsigned int img_height, std::vector<Rectangle> &allCandidates,
	ImageUnion* _img);

#ifdef __cplusplus
}

#endif

#endif
