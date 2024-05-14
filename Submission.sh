#!/bin/bash
#SBATCH -p gpu                          # Specify partition [Compute/Memory/GPU]
#SBATCH -N 1 -c 64                      # Specify number of nodes and processors per task
#SBATCH --ntasks-per-node=1		        # Specify number of tasks per node
#SBATCH --gpus-per-node=4		        # Specify total number of GPUs
#SBATCH -t 00:20:00                     # Specify maximum time limit (hour: minute: second)
#SBATCH -A lt900054                     # Specify project name
#SBATCH -J JOBNAME                      # Specify job name

ml Mamba
conda deactivate # Deactivate previous environment
conda activate /project/lt900054-ai2416/400850-Tong/env/final_env # Activate selected environment


nvidia-smi


# Print `Hello world`
srun python ./scripts/execute_query_engine_parallel.py \
    --query-json ./scripts/question_sample.json