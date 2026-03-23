cmake_minimum_required(VERSION 4.1)
project(lab2)

set(CMAKE_CXX_STANDARD 20)
find_package(OpenMP REQUIRED)


set(CMAKE_CXX_STANDARD 20)

add_executable(lab2 main.cpp)

if(OpenMP_CXX_FOUND)
    target_link_libraries(lab2 PUBLIC OpenMP::OpenMP_CXX)
endif()
