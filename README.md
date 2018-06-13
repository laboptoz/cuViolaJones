# cuViolaJones
A CUDA implementation of Viola Jones<br>
Runtime parameters can be set in the file "macros.hpp".<br>
The options are:<br>
<table>
  <tr>
    <th>Macro</th>
    <th>Value</th>
  </tr>
  <tr>
    <th>MODE</th>
    <th>0 = Single Image Detection<br>
        1 = Metric Test<br>
        2 = Webcam mode</th>
  </tr>
  <tr>
    <th>CPUMULTI</th>
    <th>1 = CPU multithreading enabled<br>
        0 = CPU multithreading disabled</th>
  </tr>
  <tr>
    <th>GPUII</th>
    <th>1 = Integral image pyramid calculation on GPU<br>
        0 = Integral image pyramid calculation on CPU</th>
  </tr>
  <tr>
    <th>TEST</th>
    <th>1 = Print integral image pyramid levels to .csv files</th>
  </tr>
  <tr>
    <th>PRINT</th>
    <th>1 = Print details during metric test</th>
  </tr>
  <tr>
    <th>REPORT_GMEM</th>
    <th>1 = Print global memory values to confirm no memory leaks</th>
  </tr>
  <tr>
    <th>SCALING</th>
    <th>The image pyramid scaling factor</th>
  </tr>
  <tr>
    <th>MIN_NEIGH</th>
    <th>Minimum neighbor rectangules used for rectangle grouping</th>
  </tr>
  <tr>
    <th>WIN_SIZE</th>
    <th>The size of the scanning window</th>
  </tr>
  <tr>
    <th>PRUNING</th>
    <th>The number of threads for warp pruning</th>
  </tr>
  <tr>
    <th>NUMIMGS</th>
    <th>Number of images to run during metric test</th>
  </tr>
  <tr>
    <th>CPUTEST</th>
    <th>If 1, runs OpenCV classifer for metric test</th>
  </tr>
  <tr>
    <th>DISPLAY</th>
    <th>If 1, displays images during metric test</th>
  </tr>
  <tr>
    <th>GPUTEST</th>
    <th>If 1, runs GPU classifier for metric test</th>
  </tr>
</table>
<br>Files are:
<table>
  <tr>
    <th>File</th>
    <th>Contents</th>  
  </tr>
  <tr>
    <th>cuViolaJones_eric.cu</th>
    <th>Runs each testing mode.</th>  
  </tr>
  <tr>
    <th>macros.hpp</th>
    <th>Holds define macros which set the testing mode.</th>  
  </tr>
  <tr>
    <th>paths.hpp</th>
    <th>Holds define macros for relative file paths.</th>  
  </tr>
  <tr>
    <th>cpuViolaJones.hpp</th>
    <th>Functions for running tests with OpenCV's cascade classifier.</th>  
  </tr>
  <tr>
    <th>gpuViolaJones.cuh</th>
    <th>Functions for running tests with our CUDA Haar classifier.</th>  
  </tr>
  <tr>
    <th>haar.cuh</th>
    <th>Functions for our CUDA Haar classifier.</th>  
  </tr>
  <tr>
    <th>haar.h</th>
    <th>Structs that are used in haar.cuh.</th>  
  </tr>
  <tr>
    <th>cuNNII_v2.cuh</th>
    <th>GPU implementation of nearest neighbor integral image pyramid.</th>  
  </tr>
  <tr>
    <th>cuda_error_check.h</th>
    <th>Error checking for CUDA function calls.</th>  
  </tr>
  <tr>
    <th>rectangles.cpp</th>
    <th>Functions for rectangle operations.</th>  
  </tr>
  <tr>
    <th>load_images.hpp</th>
    <th>Image loading functions for metric test.</th>  
  </tr>
  <tr>
    <th>parameter_loader.h</th>
    <th>Functions for loading classifier parameters from file to GPU.</th>  
  </tr>
</table>
