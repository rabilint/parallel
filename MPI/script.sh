#!/bin/bash

clear

mpicxx -std=c++20 main.cpp -o main && mpiexec --oversubscribe -n 5 ./main
