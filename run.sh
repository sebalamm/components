#!/bin/bash
./compile.sh

echo "Test Runs GNM 16"
mpirun -n 1 ./build/shortcuts   -k 8 -n 16 -m 20 -gen gnm_undirected -seq 1024 -i 1 -seed 1 | grep "^COMPONENTS\|RESULT\|INPUT" 
mpirun -n 1 ./build/local       -k 8 -n 16 -m 20 -gen gnm_undirected -seq 1024 -i 1 -seed 1 | grep "^COMPONENTS\|RESULT\|INPUT" 
mpirun -n 1 ./build/exponential -k 8 -n 16 -m 20 -gen gnm_undirected -seq 1024 -i 1 -seed 1 | grep "^COMPONENTS\|RESULT\|INPUT" 
echo "Benchmark Runs GNM 16"
mpirun -n 8 ./build/exponential -k 8 -n 16 -m 20 -gen gnm_undirected -seq 1024 -i 1 -seed 1 | grep "^COMPONENTS\|RESULT\|INPUT" 
mpirun -n 8 ./build/local       -k 8 -n 16 -m 20 -gen gnm_undirected -seq 1024 -i 1 -seed 1 | grep "^COMPONENTS\|RESULT\|INPUT" 
mpirun -n 8 ./build/shortcuts   -k 8 -n 16 -m 20 -gen gnm_undirected -seq 1024 -i 1 -seed 1 | grep "^COMPONENTS\|RESULT\|INPUT" 

echo "Test Runs RGG 16"
mpirun -n 1 ./build/shortcuts   -k 8 -n 16 -r 0.005 -gen rgg_2d -seq 1024 -i 1 -seed 1 | grep "^COMPONENTS\|RESULT\|INPUT" 
mpirun -n 1 ./build/local       -k 8 -n 16 -r 0.005 -gen rgg_2d -seq 1024 -i 1 -seed 1 | grep "^COMPONENTS\|RESULT\|INPUT" 
mpirun -n 1 ./build/exponential -k 8 -n 16 -r 0.005 -gen rgg_2d -seq 1024 -i 1 -seed 1 | grep "^COMPONENTS\|RESULT\|INPUT" 
echo "Benchmark Runs RGG 16"
mpirun -n 8 ./build/shortcuts   -k 8 -n 16 -r 0.005 -gen rgg_2d -seq 1024 -i 1 -seed 1 | grep "^COMPONENTS\|RESULT\|INPUT" 
mpirun -n 8 ./build/local       -k 8 -n 16 -r 0.005 -gen rgg_2d -seq 1024 -i 1 -seed 1 | grep "^COMPONENTS\|RESULT\|INPUT" 
mpirun -n 8 ./build/exponential -k 8 -n 16 -r 0.005 -gen rgg_2d -seq 1024 -i 1 -seed 1 | grep "^COMPONENTS\|RESULT\|INPUT" 
