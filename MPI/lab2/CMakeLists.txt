cmake_minimum_required(VERSION 4.2)
project(lab2)

set(CMAKE_CXX_STANDARD 20)
find_package(MPI REQUIRED)

add_executable(lab2 main.cpp)
target_link_libraries(lab2 PRIVATE MPI::MPI_CXX)