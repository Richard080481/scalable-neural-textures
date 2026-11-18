#!/bin/bash
# Two-node, four-A40 example. Confirm availability with noderes before submission.
# Submit: sbatch --account=<project_id> slurm/train_4gpu_carc.sbatch
#SBATCH --job-name=nt-4gpu
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=a40:2
#SBATCH --mem=64G
#SBATCH --time=00:30:00
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
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
: "${NT_OUTPUT_PREFIX:=scaling_4gpu}"
export NT_OUTPUT="outputs/${NT_OUTPUT_PREFIX}_${SLURM_JOB_ID}"
: "${NT_CONFIG:=configs/scaling.yaml}"
export NT_CONFIG
export NT_OUTPUT_PREFIX

srun --label bash -c '
  torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node=2 \
    --node_rank="$SLURM_NODEID" \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    -m neural_texture.train \
    --config "$NT_CONFIG" \
    --output-dir "$NT_OUTPUT"
'
