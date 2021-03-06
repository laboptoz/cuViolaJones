set(OpenCV_DIR "opencv/build")
cmake_minimum_required(VERSION 2.8)
project("cuViolaJones")
find_package( OpenCV REQUIRED )


include_directories(
  ${SOURCE_DIR}
  ${CUDA_TOOLKIT_INCLUDE}
  ${NVTX_INCLUDE_DIR}
  ${NVML_INCLUDE_DIR}
  ${FREEGLUT_INCLUDE_DIR}
  ${GLEW_INCLUDE_DIR}
  ${CUDA_TOOLKIT_SAMPLES_DIR}/common/inc
)

#######################################
#######################################
file(GLOB HEADERS *.h *.cuh *.hpp)
source_group("Headers" FILES ${HEADERS})

cuda_add_executable( ${PROJECT_NAME} 
${HEADERS}
#cuViolaJones_will.cu
cuViolaJones_eric.cu
rectangles.cpp
OPTIONS --cudart=shared  -lineinfo -arch sm_52 -use_fast_math
DEBUG -Xptxas -v -g -G 
)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/${OpenCV_DIR}/bin/$<CONFIG>/"
        "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
		
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/data/"
        "${CMAKE_CURRENT_BINARY_DIR}/data/"
		
	)
		
target_link_libraries( 
${PROJECT_NAME}

${CUDA_CUDA_LIBRARY}
${CUDA_CUDART_LIBRARY}
${CUDA_CURAND_LIBRARY}
${NVML_DEBUG_LIBRARIES}
${NVTX_DEBUG_LIBRARIES}
${FREEGLUT_DEBUG_LIBRARIES}
${GLEW_DEBUG_LIBRARIES}
${OpenCV_LIBS}
)