#!/bin/bash
# Submit with: sbatch --account=<project_id> slurm/train_1gpu_carc.sbatch
#SBATCH --job-name=nt-1gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=a40:1
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=outputs/slurm-%x-%j.out

set -euo pipefail
module purge
module load conda

# Create this environment once in an interactive CARC allocation, then set
# NT_ENV to its absolute path before submitting if it differs.
: "${NT_ENV:=$HOME/.conda/envs/scalable-neural-textures}"
eval "$(conda shell.bash hook)"
conda activate "$NT_ENV"

cd "$SLURM_SUBMIT_DIR"
nvidia-smi
: "${NT_CONFIG:=configs/scaling.yaml}"
: "${NT_OUTPUT_PREFIX:=scaling_1gpu}"
torchrun --standalone --nproc_per_node=1 -m neural_texture.train \
  --config "$NT_CONFIG" \
  --output-dir "outputs/${NT_OUTPUT_PREFIX}_${SLURM_JOB_ID}"

