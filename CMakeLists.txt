cmake_minimum_required (VERSION 3.8)
project ("ColorConverter" LANGUAGES CXX CUDA)
set(HEADERS ColorConvertor.h MediaHandler.h CudaHandler.cuh)
find_package(OpenCV REQUIRED PATHS C:/OpenCV/build/install/x64/vc17/lib)
find_package(CUDA REQUIRED)
aux_source_directory(
	./src/ SRC_LIST
)
include_directories(${PROJECT_NAME} ${HEADERS})
add_executable(${PROJECT_NAME} ${SRC_LIST})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} cudart)
