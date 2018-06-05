# CMake generated Testfile for 
# Source directory: ${CMAKE_CURRENT_SOURCE_DIR}/opencv/modules/videostab
# Build directory: ${CMAKE_CURRENT_SOURCE_DIR}/opencv/build/modules/videostab
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(opencv_test_videostab "${CMAKE_CURRENT_SOURCE_DIR}/opencv/build/bin/opencv_test_videostab.exe" "--gtest_output=xml:opencv_test_videostab.xml")
set_tests_properties(opencv_test_videostab PROPERTIES  LABELS "Main;opencv_videostab;Accuracy" WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/opencv/build/test-reports/accuracy")
