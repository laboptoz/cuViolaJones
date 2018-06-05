# CMake generated Testfile for 
# Source directory: C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/modules/imgcodecs
# Build directory: C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/modules/imgcodecs
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(opencv_test_imgcodecs "C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/bin/opencv_test_imgcodecs.exe" "--gtest_output=xml:opencv_test_imgcodecs.xml")
set_tests_properties(opencv_test_imgcodecs PROPERTIES  LABELS "Main;opencv_imgcodecs;Accuracy" WORKING_DIRECTORY "C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/test-reports/accuracy")
add_test(opencv_perf_imgcodecs "C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/bin/opencv_perf_imgcodecs.exe" "--gtest_output=xml:opencv_perf_imgcodecs.xml")
set_tests_properties(opencv_perf_imgcodecs PROPERTIES  LABELS "Main;opencv_imgcodecs;Performance" WORKING_DIRECTORY "C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/test-reports/performance")
add_test(opencv_sanity_imgcodecs "C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/bin/opencv_perf_imgcodecs.exe" "--gtest_output=xml:opencv_perf_imgcodecs.xml" "--perf_min_samples=1" "--perf_force_samples=1" "--perf_verify_sanity")
set_tests_properties(opencv_sanity_imgcodecs PROPERTIES  LABELS "Main;opencv_imgcodecs;Sanity" WORKING_DIRECTORY "C:/Users/ngodwin/Desktop/class_labs/Src/cuViolaJones/opencv/build/test-reports/sanity")
