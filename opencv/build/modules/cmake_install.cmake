# Install script for directory: C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/sources/modules

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/modules/.firstpass/calib3d/cmake_install.cmake")
  include("C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/modules/.firstpass/core/cmake_install.cmake")
  include("C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/modules/.firstpass/dnn/cmake_install.cmake")
  include("C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/modules/.firstpass/features2d/cmake_install.cmake")
  include("C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/modules/.firstpass/flann/cmake_install.cmake")
  include("C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/modules/.firstpass/highgui/cmake_install.cmake")
  include("C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/modules/.firstpass/imgcodecs/cmake_install.cmake")
  include("C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/modules/.firstpass/imgproc/cmake_install.cmake")
  include("C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/modules/.firstpass/java/cmake_install.cmake")
  include("C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/modules/.firstpass/js/cmake_install.cmake")
  include("C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/modules/.firstpass/ml/cmake_install.cmake")
  include("C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/modules/.firstpass/objdetect/cmake_install.cmake")
  include("C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/modules/.firstpass/photo/cmake_install.cmake")
  include("C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/modules/.firstpass/python/cmake_install.cmake")
  include("C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/modules/.firstpass/shape/cmake_install.cmake")
  include("C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/modules/.firstpass/stitching/cmake_install.cmake")
  include("C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/modules/.firstpass/superres/cmake_install.cmake")
  include("C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/modules/.firstpass/ts/cmake_install.cmake")
  include("C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/modules/.firstpass/video/cmake_install.cmake")
  include("C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/modules/.firstpass/videoio/cmake_install.cmake")
  include("C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/modules/.firstpass/videostab/cmake_install.cmake")
  include("C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/modules/.firstpass/viz/cmake_install.cmake")
  include("C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/modules/.firstpass/world/cmake_install.cmake")
  include("C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/modules/core/cmake_install.cmake")
  include("C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/modules/imgproc/cmake_install.cmake")
  include("C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/modules/objdetect/cmake_install.cmake")
  include("C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/modules/imgcodecs/cmake_install.cmake")
  include("C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/modules/highgui/cmake_install.cmake")

endif()

