#!/bin/bash
# Submit: sbatch slurm/setup_carc.sbatch
#SBATCH --job-name=nt-setup
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/slurm-%x-%j.out

set -euo pipefail
if ! command -v module >/dev/null 2>&1; then
  source /etc/profile.d/modules.sh
fi
module purge
module load conda
eval "$(conda shell.bash hook)"

: "${NT_ENV:=$HOME/.conda/envs/scalable-neural-textures}"
if [[ ! -d "$NT_ENV" ]]; then
  conda create -p "$NT_ENV" python=3.11 -y
fi
conda activate "$NT_ENV"

cd "$SLURM_SUBMIT_DIR"
python -m pip install -e '.[dev]'
pytest
python -c "import torch; print('PyTorch:', torch.__version__)"
