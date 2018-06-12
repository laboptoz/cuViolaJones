#define MODE 2			// 0: Single image detection, 1: Run metric test, 2: Webcam mode

#define CPUMULTI 1		//use cpu multithreadings

#define GPUII 1			//use gpu integral image
#if GPUII
	#define TEST 0			//print test information
#endif

#define PRINT 0			// Output details?
#define REPORT_GMEM 1

// Parameters for both CPU and GPU
#define SCALING 1.2
#define MIN_NEIGH 1
#define WIN_SIZE 24
#define PRUNING 32


//Parameters for run metric test
#define NUMIMGS 1000
#define IMGDELAY 3		//not used
#define CPUTEST 0
#define DISPLAY 0       // Display images?
#define GPUTEST 1