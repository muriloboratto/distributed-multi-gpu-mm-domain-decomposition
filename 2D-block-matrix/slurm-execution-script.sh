#!/bin/bash
#SBATCH --partition=sequana_gpu_dev
#SBATCH --account=treinamento
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus=4

module load nvshmem/3.1.7_cuda-11.2_sequana
module load anaconda3/2020.07_sequana 

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

python3 plot.py 