cmake_minimum_required(VERSION 4.2)
project(lab4)

set(CMAKE_CXX_STANDARD 20)
find_package(MPI REQUIRED)

add_executable(lab4 main.cpp)
target_link_libraries(lab4 PRIVATE MPI::MPI_CXX)
