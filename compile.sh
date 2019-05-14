#!/bin/bash

mkdir build
cd build
cmake -DCMAKE_C_COMPILER=gcc-9 -DCMAKE_CXX_COMPILER=g++-9 ../
make -j4
