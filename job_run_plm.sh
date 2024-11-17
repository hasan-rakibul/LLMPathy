#!/bin/bash
 
#SBATCH --job-name=Run-PLM
#SBATCH --output=log_slurm/%x_%j.out
#SBATCH --error=log_slurm/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=pawsey1001-gpu

export TOKENIZERS_PARALLELISM=false
python src/run_plm.py
