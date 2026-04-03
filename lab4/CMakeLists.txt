cmake_minimum_required(VERSION 4.1)

project(lab4)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
find_package(OpenMP REQUIRED)

add_executable(lab4 main.cpp)

if(OpenMP_CXX_FOUND)
    target_link_libraries(lab4 PUBLIC OpenMP::OpenMP_CXX)
endif()

target_link_libraries(lab4 PUBLIC lapacke openblas)