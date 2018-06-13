#define MODE 1			// 0: Single image detection, 1: Run metric test, 2: Webcam mode

#define CPUMULTI 1		//use cpu multithreadings

#define GPUII 1			//use gpu integral image
#if GPUII
	#define TEST 0			//print test information
#endif

#define PRINT 0			// Output details?
#define REPORT_GMEM 0

// Parameters for both CPU and GPU
#define SCALING 1.2
#define MIN_NEIGH 1
#define WIN_SIZE 24
#define PRUNING 9


//Parameters for run metric test
#define NUMIMGS 1000
#define IMGDELAY 0		//not used
#define CPUTEST 1
#define DISPLAY 1       // Display images?
#define GPUTEST 1
