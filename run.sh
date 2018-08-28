#!/bin/bash
./compile.sh

# nonexclusive mpirun -n 32 ./build/components -n 20 -m 20 -gen gnm_undirected -seq 1024 -i 1 -seed 1 | grep "^COMPONENTS\|RESULT" 
# nonexclusive mpirun -n 32 ./build/shortcuts -n 20 -m 20 -gen gnm_undirected -seq 1024 -i 1 -seed 1 | grep "^COMPONENTS\|RESULT" 

echo "Test Runs"
mpirun -n 1 --oversubscribe ./build/components -k 16 -n 20 -m 24 -gen gnm_undirected -seq 1024 -i 1 -seed 1 | grep "^COMPONENTS\|RESULT\|INPUT" 
mpirun -n 1 --oversubscribe ./build/shortcuts -k 16 -n 20 -m 24 -gen gnm_undirected -seq 1024 -i 1 -seed 1 | grep "^COMPONENTS\|RESULT\|INPUT" 
echo "Benchmark Runs"
mpirun -n 16 --oversubscribe ./build/components -k 16 -n 20 -m 24 -gen gnm_undirected -seq 1024 -i 1 -seed 1 | grep "^COMPONENTS\|RESULT\|INPUT" 
mpirun -n 16 --oversubscribe ./build/shortcuts -k 16 -n 20 -m 24 -gen gnm_undirected -seq 1024 -i 1 -seed 1 | grep "^COMPONENTS\|RESULT\|INPUT" 

echo "Test Runs"
mpirun -n 1 --oversubscribe ./build/components -k 16 -n 20 -r 0.0013 -gen rgg_2d -seq 1024 -i 1 -seed 1 | grep "^COMPONENTS\|RESULT\|INPUT" 
mpirun -n 1 --oversubscribe ./build/shortcuts -k 16 -n 20 -r 0.0013 -gen rgg_2d -seq 1024 -i 1 -seed 1 | grep "^COMPONENTS\|RESULT\|INPUT" 
echo "Benchmark Runs"
mpirun -n 16 --oversubscribe ./build/components -k 16 -n 20 -r 0.0013 -gen rgg_2d -seq 1024 -i 1 -seed 1 | grep "^COMPONENTS\|RESULT\|INPUT" 
mpirun -n 16 --oversubscribe ./build/shortcuts -k 16 -n 20 -r 0.0013 -gen rgg_2d -seq 1024 -i 1 -seed 1 | grep "^COMPONENTS\|RESULT\|INPUT" 
