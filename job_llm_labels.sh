#!/bin/bash
 
#SBATCH --job-name=LLM-Labels
#SBATCH --output=log_slurm/%j_%x.out
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --account=pawsey1001

. .declare_api_key.sh

python src/llm_labels.py \
--file_path=data/NewsEmp2022/messages_dev_features_ready_for_WS_2022_llama.tsv
