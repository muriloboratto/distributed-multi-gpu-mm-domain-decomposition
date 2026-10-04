#!/bin/bash

# SUMMA comparison:
#   MMM = MPI-SYNC           
#   YYY = MPI-CHUNKED-ASYNC
#   CCC = CUDA-Aware MPI-SYNC 
#   XXX = CUDA-Aware MPI-ASYNC (GPU-direct MPI_Ibcast + chunked B/GEMM)
#   NNN = NCCL-ASYNC (row/column broadcasts + two-stream B/GEMM pipeline)
#   WWW = NVSHMEM-SYNC (blocking full-panel GET A/B -> monolithic GEMM)
#   SSS = NVSHMEM-CHUNKED-ASYNC (nonblocking GET + double-buffered B/GEMM overlap)

for i in 2048 4096 8192 16384 32768
do
    for lib in MMM YYY CCC XXX NNN WWW SSS
    do
        echo "Matrix size = $i | Library = $lib" | tee -a result--2048-32768-1node-4GPUs-${lib}.txt

        mpirun -np 1 ./mmb 0 $i $lib : \
               -np 1 ./mmb 1 $i $lib : \
               -np 1 ./mmb 2 $i $lib : \
               -np 1 ./mmb 3 $i $lib \
               >> result--2048-32768-1node-4GPUs-${lib}.txt
    done
    echo "-----------"
done
