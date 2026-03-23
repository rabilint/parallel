cmake_minimum_required(VERSION 4.1)
project(lab3)

find_package(OpenMP REQUIRED)
set(CMAKE_CXX_STANDARD 20)


set(CMAKE_CXX_STANDARD 20)
add_executable(lab3 main.cpp)

if(OpenMP_CXX_FOUND)
    target_link_libraries(lab3 PUBLIC OpenMP::OpenMP_CXX)
endif()