#!/bin/bash
# Submit: sbatch slurm/showcase_1gpu_carc.sbatch
#SBATCH --job-name=nt-showcase
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=a40:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/slurm-%x-%j.out

set -euo pipefail
module purge
module load conda
: "${NT_ENV:=$HOME/.conda/envs/scalable-neural-textures}"
eval "$(conda shell.bash hook)"
conda activate "$NT_ENV"
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
nvidia-smi
torchrun --standalone --nproc_per_node=1 \
  -m neural_texture.train \
  --config configs/showcase.yaml \
  --output-dir "outputs/showcase_1gpu_${SLURM_JOB_ID}"
