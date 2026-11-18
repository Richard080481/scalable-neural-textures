# Running on USC CARC

The target is Discovery's `gpu` partition. Before choosing a device, inspect
current capacity with `noderes -c -g` and availability with `noderes -f -g`.
The included single-GPU job requests one A40. Change the GPU type only after
checking CARC availability; current documented types include L40S, A100, A40,
V100, and P100, with at most two GPUs per node for A100/A40/V100/P100 and three
L40S GPUs per node.

Create an environment from an interactive allocation rather than on a login
node:

```bash
salloc --partition=gpu --ntasks=1 --gpus-per-task=a40:1 --cpus-per-task=8 --mem=32G --time=01:00:00
module purge
module load conda
eval "$(conda shell.bash hook)"
conda create -p "$HOME/.conda/envs/scalable-neural-textures" python=3.11 -y
conda activate "$HOME/.conda/envs/scalable-neural-textures"
python -m pip install -e '.[dev]'
pytest
```

Submit identical strong-scaling workloads. Replace `<project_id>` and record
the returned job IDs:

```bash
mkdir -p outputs
sbatch slurm/train_1gpu_carc.sbatch
sbatch slurm/train_2gpu_carc.sbatch
sbatch slurm/train_4gpu_carc.sbatch
```

The four-GPU template uses two nodes with two A40s each. Discovery's generally
available inventory does not provide four A40/A100/V100/P100 GPUs on one node.
If your project has suitable Endeavour resources, adjust the partition and
topology while keeping all GPUs identical. Never mix hardware in one curve.

Run the no-sync comparison on two GPUs by overriding only the config:

```bash
sbatch --export=ALL,NT_CONFIG=configs/accumulation_baseline.yaml slurm/train_2gpu_carc.sbatch
sbatch --export=ALL,NT_CONFIG=configs/optimized.yaml slurm/train_2gpu_carc.sbatch
```

After the jobs finish, create presentation graphics on a login node (this is a
lightweight post-processing task):

```bash
python scripts/plot_results.py \
  outputs/scaling_1gpu_<jobid> \
  outputs/scaling_2gpu_<jobid> \
  outputs/scaling_4gpu_<jobid> \
  --output-dir outputs/plots
```

Each run already contains `renders/training_progress.gif`, PNG frames,
`metrics.csv`, `summary.json`, and checkpoints. Enable `profiling.enabled` in a
copied config to emit per-rank traces under `profiler/`.

Official references:

- https://www.carc.usc.edu/user-guides/advanced-hpc-programming/gpu-programming
- https://www.carc.usc.edu/user-guides/hpc-systems/using-our-hpc-systems/running-jobs
- https://www.carc.usc.edu/user-guides/data-science/building-conda-environment

