#!/bin/bash
 
#SBATCH --job-name=TUNE-HPARAMS
#SBATCH --output=log_slurm/%x_%j.out
#SBATCH --error=log_slurm/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu-highmem
#SBATCH --gres=gpu:1
#SBATCH --account=pawsey1001-gpu

export TOKENIZERS_PARALLELISM=false
python src/optuna_tune_hparams.py
