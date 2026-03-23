cmake_minimum_required(VERSION 4.1)
project(project)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenMP REQUIRED)

add_executable(project main.cpp)

if(OpenMP_CXX_FOUND)
    target_link_libraries(project PUBLIC OpenMP::OpenMP_CXX)
endif()