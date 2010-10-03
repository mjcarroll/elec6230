#!/bin/bash

USER=aubcls17
mkdir -p /scratch/$USER

cp h5parallel /scratch/$USER
cd /scratch/$USER
for chunk in 10 100 200 500
do
    echo Chunk: $chunk 
    for nthread in 1 2 3 4 5 6
    do 
        echo Threads: $nthread
        ./h5parallel $nthread $chunk>> results
    done
done
