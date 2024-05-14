#!/bin/bash
#SBATCH -p gpu                          # Specify partition [Compute/Memory/GPU]
#SBATCH -N 1 -c 64                      # Specify number of nodes and processors per task
#SBATCH --ntasks-per-node=1		        # Specify number of tasks per node
#SBATCH --gpus-per-node=4		        # Specify total number of GPUs
#SBATCH -t 12:00:00                    # Specify maximum time limit (hour: minute: second)
#SBATCH -A lt900047                     # Specify project name
#SBATCH -J TBLC_Pangpuriye_hackathon_submission         # Specify job name
#SBATCH -o /home/projratc/SPAIHack/logs/TBLC-Pangpuriye.out


ml Mamba
conda deactivate # Deactivate previous environment
conda activate /project/lt900054-ai2416/env/prod # Activate selected environment

srun python /project/lt900054-ai2416/400850-Tong/table-query-engine/scripts/execute_query_engine_C.py \
    --query-json /home/projratc/SPAIHack/TBLC/final_question.json \
    --save-dir /home/projratc/SPAIHack/submission/TBLC/Pangpuriye-final-submission.jsonl

