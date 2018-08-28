#!/bin/bash
./compile.sh
nonexclusive mpirun -n 32 ./build/components -size 20 -seq 1024 -i 1 -seed 1 | tee >(grep "^COMPONENTS|^RESULT")
nonexclusive mpirun -n 32 ./build/shortcuts -size 20 -seq 1024 -i 1 -seed 1 | tee >(grep "^COMPONENTS|^RESULT")
