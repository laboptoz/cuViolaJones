# Install script for directory: ${CMAKE_CURRENT_SOURCE_DIR}/opencv/include

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Program Files/OpenCV")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
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

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv" TYPE FILE FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/opencv/include/opencv/cv.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/opencv/include/opencv/cv.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/opencv/include/opencv/cvaux.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/opencv/include/opencv/cvaux.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/opencv/include/opencv/cvwimage.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/opencv/include/opencv/cxcore.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/opencv/include/opencv/cxcore.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/opencv/include/opencv/cxeigen.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/opencv/include/opencv/cxmisc.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/opencv/include/opencv/highgui.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/opencv/include/opencv/ml.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2" TYPE FILE FILES "${CMAKE_CURRENT_SOURCE_DIR}/opencv/include/opencv2/opencv.hpp")
endif()

