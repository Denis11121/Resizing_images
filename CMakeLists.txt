cmake_minimum_required(VERSION 3.15)
project(proiect)

set(CMAKE_CXX_STANDARD 17)

# Add OpenCV paths
set(OpenCV_DIR "D:/opencv/opencv/build/x64/vc16/lib/")

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(proiect main.cpp src/proiect.cpp)

# Link OpenCV libraries to the executable
target_link_libraries(proiect ${OpenCV_LIBS})