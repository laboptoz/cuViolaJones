# Install script for directory: C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv

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
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2" TYPE FILE FILES "C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/cvconfig.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv2" TYPE FILE FILES "C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/opencv2/opencv_modules.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/x64/vc14/lib/OpenCVModules.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/x64/vc14/lib/OpenCVModules.cmake"
         "C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/CMakeFiles/Export/x64/vc14/lib/OpenCVModules.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/x64/vc14/lib/OpenCVModules-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/x64/vc14/lib/OpenCVModules.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/x64/vc14/lib" TYPE FILE FILES "C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/CMakeFiles/Export/x64/vc14/lib/OpenCVModules.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^()$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/x64/vc14/lib" TYPE FILE FILES "C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/CMakeFiles/Export/x64/vc14/lib/OpenCVModules-noconfig.cmake")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/x64/vc14/lib" TYPE FILE FILES
    "C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/win-install/OpenCVConfig-version.cmake"
    "C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/win-install/x64/vc14/lib/OpenCVConfig.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/." TYPE FILE FILES
    "C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/win-install/OpenCVConfig-version.cmake"
    "C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/win-install/OpenCVConfig.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xlibsx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/." TYPE FILE PERMISSIONS OWNER_READ GROUP_READ WORLD_READ FILES "C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/LICENSE")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/3rdparty/zlib/cmake_install.cmake")
  include("C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/3rdparty/libjpeg/cmake_install.cmake")
  include("C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/3rdparty/libtiff/cmake_install.cmake")
  include("C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/3rdparty/libwebp/cmake_install.cmake")
  include("C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/3rdparty/libjasper/cmake_install.cmake")
  include("C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/3rdparty/libpng/cmake_install.cmake")
  include("C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/3rdparty/openexr/cmake_install.cmake")
  include("C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/3rdparty/ippiw/cmake_install.cmake")
  include("C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/3rdparty/protobuf/cmake_install.cmake")
  include("C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/3rdparty/ittnotify/cmake_install.cmake")
  include("C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/include/cmake_install.cmake")
  include("C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/modules/cmake_install.cmake")
  include("C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/doc/cmake_install.cmake")
  include("C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/data/cmake_install.cmake")
  include("C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/apps/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
