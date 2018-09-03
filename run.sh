#!/bin/bash
./compile.sh

echo "Test Runs GNM 20"
nonexclusive mpirun -n 1 ./build/components -k 16 -n 20 -m 24 -gen gnm_undirected -seq 1024 -i 1 -seed 1 | grep "^COMPONENTS\|RESULT\|INPUT" 
echo "Benchmark Runs GNM 20"
nonexclusive mpirun -n 16 ./build/components -k 16 -n 20 -m 24 -gen gnm_undirected -seq 1024 -i 1 -seed 1 | grep "^COMPONENTS\|RESULT\|INPUT" 
nonexclusive mpirun -n 16 ./build/shortcuts -k 16 -n 20 -m 24 -gen gnm_undirected -seq 1024 -i 1 -seed 1 | grep "^COMPONENTS\|RESULT\|INPUT" 

echo "Test Runs RGG 20"
nonexclusive mpirun -n 1 ./build/components -k 16 -n 20 -r 0.0013 -gen rgg_2d -seq 1024 -i 1 -seed 1 | grep "^COMPONENTS\|RESULT\|INPUT" 
echo "Benchmark Runs RGG 20"
nonexclusive mpirun -n 16 ./build/components -k 16 -n 20 -r 0.0013 -gen rgg_2d -seq 1024 -i 1 -seed 1 | grep "^COMPONENTS\|RESULT\|INPUT" 
nonexclusive mpirun -n 16 ./build/shortcuts -k 16 -n 20 -r 0.0013 -gen rgg_2d -seq 1024 -i 1 -seed 1 | grep "^COMPONENTS\|RESULT\|INPUT" 
