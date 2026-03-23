cmake_minimum_required(VERSION 4.1)
project(lab1)

set(CMAKE_CXX_STANDARD 20)
find_package(OpenMP REQUIRED)

add_executable(lab1 main.cpp)

if(OpenMP_CXX_FOUND)
    target_link_libraries(lab1 PUBLIC OpenMP::OpenMP_CXX)
endif()
