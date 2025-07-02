#!/bin/bash
 
#SBATCH --job-name=Test-PLM
#SBATCH --output=log_slurm/%j_%x.out
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=pawsey1001-gpu

module load pytorch/2.2.0-rocm5.7.3

# Normal test of PLM
# singularity exec $SINGULARITY_CONTAINER bash -c "\
# source $MYSOFTWARE/.venv/bin/activate && \
# export TOKENIZERS_PARALLELISM=false && \
# python src/test.py"

# Zero-shot test of LLM
singularity exec $SINGULARITY_CONTAINER bash -c "\
source $MYSOFTWARE/.venv/bin/activate && \
export TOKENIZERS_PARALLELISM=false && \
python src/test.py --config=config/config_test_zero_shot.yaml"
