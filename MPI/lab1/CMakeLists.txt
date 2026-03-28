cmake_minimum_required(VERSION 4.2)
project(lab1)

set(CMAKE_CXX_STANDARD 20)
find_package(MPI REQUIRED)

add_executable(lab1 main.cpp
)
target_link_libraries(lab1 PRIVATE MPI::MPI_CXX)
