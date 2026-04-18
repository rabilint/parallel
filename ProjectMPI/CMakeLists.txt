cmake_minimum_required(VERSION 4.2)
project(ProjectMPI)

set(CMAKE_CXX_STANDARD 20)
find_package(MPI REQUIRED)

add_executable(ProjectMPI src/main.cpp
        src/generator.cpp
        include/generator.h
        include/analyzer.h
        src/analyzer.cpp
        include/mpi_io.h
        src/mpi_io.cpp
)
target_link_libraries(ProjectMPI PRIVATE MPI::MPI_CXX)