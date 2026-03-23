cmake_minimum_required(VERSION 4.1)
project(test)

#set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)
find_package(OpenMP REQUIRED)


add_executable(test test.cpp)
if(OpenMP_CXX_FOUND)
    target_link_libraries(test PUBLIC OpenMP::OpenMP_CXX)
endif()

#
#add_executable(test main.c
#        test.cpp)
