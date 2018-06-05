# CMake generated Testfile for 
# Source directory: ${CMAKE_CURRENT_SOURCE_DIR}/opencv/modules/objdetect
# Build directory: ${CMAKE_CURRENT_SOURCE_DIR}/opencv/build/modules/objdetect
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(opencv_test_objdetect "${CMAKE_CURRENT_SOURCE_DIR}/opencv/build/bin/opencv_test_objdetect.exe" "--gtest_output=xml:opencv_test_objdetect.xml")
set_tests_properties(opencv_test_objdetect PROPERTIES  LABELS "Main;opencv_objdetect;Accuracy" WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/opencv/build/test-reports/accuracy")
add_test(opencv_perf_objdetect "${CMAKE_CURRENT_SOURCE_DIR}/opencv/build/bin/opencv_perf_objdetect.exe" "--gtest_output=xml:opencv_perf_objdetect.xml")
set_tests_properties(opencv_perf_objdetect PROPERTIES  LABELS "Main;opencv_objdetect;Performance" WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/opencv/build/test-reports/performance")
add_test(opencv_sanity_objdetect "${CMAKE_CURRENT_SOURCE_DIR}/opencv/build/bin/opencv_perf_objdetect.exe" "--gtest_output=xml:opencv_perf_objdetect.xml" "--perf_min_samples=1" "--perf_force_samples=1" "--perf_verify_sanity")
set_tests_properties(opencv_sanity_objdetect PROPERTIES  LABELS "Main;opencv_objdetect;Sanity" WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/opencv/build/test-reports/sanity")
