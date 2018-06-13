# cuViolaJones
A CUDA implementation of Viola Jones

Files are:
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
