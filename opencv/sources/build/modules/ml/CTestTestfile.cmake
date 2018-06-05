# CMake generated Testfile for 
# Source directory: ${CMAKE_CURRENT_SOURCE_DIR}/opencv/modules/ml
# Build directory: ${CMAKE_CURRENT_SOURCE_DIR}/opencv/build/modules/ml
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(opencv_test_ml "${CMAKE_CURRENT_SOURCE_DIR}/opencv/build/bin/opencv_test_ml.exe" "--gtest_output=xml:opencv_test_ml.xml")
set_tests_properties(opencv_test_ml PROPERTIES  LABELS "Main;opencv_ml;Accuracy" WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/opencv/build/test-reports/accuracy")
