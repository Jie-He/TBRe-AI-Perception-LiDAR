cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(perception)

find_package(PCL 1.2 REQUIRED)
find_package(OpenCV REQUIRED) 

include_directories(${PCL_INCLUDE_DIRS})
include_directories( ${OpenCV_INCLUDE_DIRS} )
#include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (perception src/perception.cpp)
target_link_libraries (perception ${PCL_LIBRARIES} ${OpenCV_LIBS})