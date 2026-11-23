#!/bin/bash
# Submit: sbatch slurm/train_2gpu_carc.sbatch
#SBATCH --job-name=nt-2gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=a40:2
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=outputs/slurm-%x-%j.out

set -euo pipefail
if ! command -v module >/dev/null 2>&1; then
  source /etc/profile.d/modules.sh
fi
module purge
module load conda
: "${NT_ENV:=$HOME/.conda/envs/scalable-neural-textures}"
eval "$(conda shell.bash hook)"
conda activate "$NT_ENV"
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / 2))
export NCCL_DEBUG=WARN
nvidia-smi
: "${NT_CONFIG:=configs/scaling.yaml}"
: "${NT_OUTPUT_PREFIX:=scaling_2gpu}"
torchrun --standalone --nproc_per_node=2 \
  -m neural_texture.train --config "$NT_CONFIG" \
  --output-dir "outputs/${NT_OUTPUT_PREFIX}_${SLURM_JOB_ID}"

