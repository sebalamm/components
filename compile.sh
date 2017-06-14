#!/bin/bash

mkdir build
cd build
cmake ..
make

cp compile_commands.json ..
