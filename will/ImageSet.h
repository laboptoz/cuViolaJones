#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <cstring>

class Image {
	public:
		unsigned char * bw;
};

class ImageSet {
	private:
		char * path = nullptr;
		Image * imgs;
	public:
		unsigned int size = 0;

		ImageSet() {}

		ImageSet(char * path) {
			this->path = path;
		}

		void loadSet(char * path) {
			this->path = path;
			loadSet();
		}

		void loadSet() {
			if (path != nullptr) {
				printf("Loading set from directory %s\n", path);
				for img in 
			}else{
				fprintf(stderr, "Error: No path provided.\n");
			}
		}

		Image * override[](unsigned int idx) {
			if (size > 0) {
				return imgs[idx];
			}else{
				fprintf("Error. Images not loaded.\n");
			}
		}
};
