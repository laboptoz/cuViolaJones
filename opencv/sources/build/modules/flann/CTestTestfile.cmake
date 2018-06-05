# CMake generated Testfile for 
# Source directory: ${CMAKE_CURRENT_SOURCE_DIR}/opencv/modules/flann
# Build directory: ${CMAKE_CURRENT_SOURCE_DIR}/opencv/build/modules/flann
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(opencv_test_flann "${CMAKE_CURRENT_SOURCE_DIR}/opencv/build/bin/opencv_test_flann.exe" "--gtest_output=xml:opencv_test_flann.xml")
set_tests_properties(opencv_test_flann PROPERTIES  LABELS "Main;opencv_flann;Accuracy" WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/opencv/build/test-reports/accuracy")
