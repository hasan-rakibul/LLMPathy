#!/bin/bash
 
#SBATCH --job-name=TUNE-HPARAMS
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=pawsey1001-gpu

export TOKENIZERS_PARALLELISM=false
python src/tune_hparams.py
