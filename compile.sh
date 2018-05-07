#!/bin/bash

mkdir build
cd build
cmake -DCMAKE_C_COMPILER=gcc-6 -DCMAKE_CXX_COMPILER=g++-6 ../
make -j4
