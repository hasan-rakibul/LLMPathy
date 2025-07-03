#!/bin/bash
 
#SBATCH --job-name=Train-PLM
#SBATCH --output=log_slurm/%j_%x.out
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=pawsey1001-gpu

module load pytorch/2.2.0-rocm5.7.3

# Train - Baseline
# singularity exec $SINGULARITY_CONTAINER bash -c "\
# source $MYSOFTWARE/.venv/bin/activate && \
# export TOKENIZERS_PARALLELISM=false && \
# python src/train_plm.py --config=config/train_base.yaml"

# Train - Noise Mitigation
singularity exec $SINGULARITY_CONTAINER bash -c "\
source $MYSOFTWARE/.venv/bin/activate && \
export TOKENIZERS_PARALLELISM=false && \
python src/train_plm.py --config=config/train_noise_mitigation.yaml"

# # Train - Additional Labels
# singularity exec $SINGULARITY_CONTAINER bash -c "\
# source $MYSOFTWARE/.venv/bin/activate && \
# export TOKENIZERS_PARALLELISM=false && \
# python src/train_plm.py --config=config/train_additional_labels.yaml"

# Train - Giorgi2024Findings
# singularity exec $SINGULARITY_CONTAINER bash -c "\
# source $MYSOFTWARE/.venv/bin/activate && \
# export TOKENIZERS_PARALLELISM=false && \
# python src/train_plm.py --config=config/Giorgi2024Findings.yaml"
