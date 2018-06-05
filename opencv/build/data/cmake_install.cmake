# Install script for directory: ${CMAKE_CURRENT_SOURCE_DIR}/opencv/sources/data

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/opencv/build/install")
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

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xlibsx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/etc/haarcascades" TYPE FILE FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/opencv/sources/data/haarcascades/haarcascade_eye.xml"
    "${CMAKE_CURRENT_SOURCE_DIR}/opencv/sources/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml"
    "${CMAKE_CURRENT_SOURCE_DIR}/opencv/sources/data/haarcascades/haarcascade_frontalcatface.xml"
    "${CMAKE_CURRENT_SOURCE_DIR}/opencv/sources/data/haarcascades/haarcascade_frontalcatface_extended.xml"
    "${CMAKE_CURRENT_SOURCE_DIR}/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml"
    "${CMAKE_CURRENT_SOURCE_DIR}/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml"
    "${CMAKE_CURRENT_SOURCE_DIR}/opencv/sources/data/haarcascades/haarcascade_frontalface_alt_tree.xml"
    "${CMAKE_CURRENT_SOURCE_DIR}/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml"
    "${CMAKE_CURRENT_SOURCE_DIR}/opencv/sources/data/haarcascades/haarcascade_fullbody.xml"
    "${CMAKE_CURRENT_SOURCE_DIR}/opencv/sources/data/haarcascades/haarcascade_lefteye_2splits.xml"
    "${CMAKE_CURRENT_SOURCE_DIR}/opencv/sources/data/haarcascades/haarcascade_licence_plate_rus_16stages.xml"
    "${CMAKE_CURRENT_SOURCE_DIR}/opencv/sources/data/haarcascades/haarcascade_lowerbody.xml"
    "${CMAKE_CURRENT_SOURCE_DIR}/opencv/sources/data/haarcascades/haarcascade_profileface.xml"
    "${CMAKE_CURRENT_SOURCE_DIR}/opencv/sources/data/haarcascades/haarcascade_righteye_2splits.xml"
    "${CMAKE_CURRENT_SOURCE_DIR}/opencv/sources/data/haarcascades/haarcascade_russian_plate_number.xml"
    "${CMAKE_CURRENT_SOURCE_DIR}/opencv/sources/data/haarcascades/haarcascade_smile.xml"
    "${CMAKE_CURRENT_SOURCE_DIR}/opencv/sources/data/haarcascades/haarcascade_upperbody.xml"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xlibsx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/etc/lbpcascades" TYPE FILE FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/opencv/sources/data/lbpcascades/lbpcascade_frontalcatface.xml"
    "${CMAKE_CURRENT_SOURCE_DIR}/opencv/sources/data/lbpcascades/lbpcascade_frontalface.xml"
    "${CMAKE_CURRENT_SOURCE_DIR}/opencv/sources/data/lbpcascades/lbpcascade_frontalface_improved.xml"
    "${CMAKE_CURRENT_SOURCE_DIR}/opencv/sources/data/lbpcascades/lbpcascade_profileface.xml"
    "${CMAKE_CURRENT_SOURCE_DIR}/opencv/sources/data/lbpcascades/lbpcascade_silverware.xml"
    )
endif()

