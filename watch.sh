#!/bin/bash
#SBATCH -p gpu                          # Specify partition [Compute/Memory/GPU]
#SBATCH -N 1 -c 64                      # Specify number of nodes and processors per task
#SBATCH --ntasks-per-node=1		        # Specify number of tasks per node
#SBATCH --gpus-per-node=4		        # Specify total number of GPUs
#SBATCH -t 00:20:00                     # Specify maximum time limit (hour: minute: second)
#SBATCH -A lt900054                     # Specify project name
#SBATCH -J JOBNAME                      # Specify job name
#SBATCH --nodelist=lanta-g-103

echo "hello test"
watch -n 1 nvidia-smi