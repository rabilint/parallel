#!/bin/bash

clear
mpicxx -std=c++20 src/main.cpp src/analyzer.cpp src/generator.cpp -Iinclude -O3 -o main
if [ $? -ne 0 ]; then
    echo "Compilation failed."
    exit 1
fi
echo "Compilation succeeded."
echo "Generating 10 million log lines..."
mpiexec -n 1 ./main -gen 10000000

echo -e "\n----------------------------------\n"
echo "Running sequential analysis..."
mpiexec -n 1 ./main -seq

echo -e "\n----------------------------------\n"
echo "Running parallel analysis with 3 processes..."
mpiexec --oversubscribe -n 3 ./main

echo -e "\n----------------------------------\n"
echo "Running parallel analysis with 6 processes..."
mpiexec --oversubscribe -n 6 ./main

echo -e "\n----------------------------------\n"
echo "Running parallel analysis with 9 processes..."
mpiexec --oversubscribe -n 9 ./main
