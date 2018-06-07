#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <string.h>
#include <iostream> 
#include <fstream>

using namespace std;
using namespace cv;

struct Image {
	Mat image;
	int x;
	int y;
	int w;
	int h;
};

/* 
*	Reads the text file that has the image names and bbox 
*	Returns an array of images and the bboxes
*/
Image * loadData(string textFile, string imagePath, int *numImgs) {
	
	ifstream labels(textFile);

	// new lines will be skipped unless we stop it from happening:    
	labels.unsetf(ios_base::skipws);

	// count the newlines with an algorithm specialized for counting:
	int line_count = count(
		istream_iterator<char>(labels),
		istream_iterator<char>(),
		'\n');

	cout << "Number of images: " << line_count << "\n" << endl;

	// Store total number of images
	*numImgs = line_count; 

	// Return to beginning of file
	labels.clear();
	labels.seekg(0, ios::beg);

	Image *imgs = new Image[line_count];
	string line;
	int idx = 0;
	while (getline(labels, line)) {
		istringstream iss(line);
		vector<string> results(istream_iterator<string>{iss}, istream_iterator<string>()); // splits line with space delimiter
		imgs[idx].image = imread(imagePath + results[0], 1); // read and store image
		imgs[idx].x = stoi(results[1]);                      // stoi() converts string to integer
		imgs[idx].y = stoi(results[2]);
		imgs[idx].w = stoi(results[3]);
		imgs[idx].h = stoi(results[4]);
		idx++;
	}

	return imgs;
}