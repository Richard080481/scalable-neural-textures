# Scalable Neural Textures

This project studies how neural-texture size and UV visibility affect gradient
communication and multi-GPU scaling. The goal is a reproducible scientific
workload and scaling analysis, not a new NeRF or rendering architecture.

An RGB texture directly stores `RGB = T(u,v)`. A neural texture stores a
learned feature vector `z = T_theta(u,v)`, sampled with differentiable bilinear
filtering; a compact CNN maps that feature image to RGB. Unlike an implicit
coordinate representation, the learnable signal is an explicit 2D feature
array rather than an MLP evaluated at each coordinate.

```text
UV [B,H,W,2] -> grid_sample(learnable [1,C,Ht,Wt]) -> feature image
                                                        |
mask ---------------------------------------------------+-> small CNN -> RGB
```

## Current status

The repository provides synthetic and directory datasets, tested UV sampling,
a configurable texture and renderer, masked L1 training, PSNR evaluation,
checkpoint resume/load, DDP, fixed-global-batch strong scaling, communication
timing, UV visibility analysis, `no_sync()` accumulation, profiler traces, and
automatic visual artifacts. No GPU scaling result is claimed until CARC jobs
have been run repeatedly on identical hardware.

## Install and test

```bash
python -m venv .venv
.venv/Scripts/activate              # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pytest
```

## Train and evaluate

```bash
python -m neural_texture.train --config configs/smoke_test.yaml --output-dir outputs/smoke
python -m neural_texture.evaluate --config outputs/smoke/config.yaml \
  --checkpoint outputs/smoke/checkpoints/step_000030.pt --output-dir outputs/smoke/renders
```

Resume a run while preserving optimizer state (the configured final step must be
greater than the checkpoint step):

```bash
python -m neural_texture.train --config configs/baseline.yaml \
  --resume outputs/previous/checkpoints/step_000100.pt --output-dir outputs/resumed
```

To materialize toy files matching the real-data interface:

```bash
python scripts/create_synthetic_dataset.py --output-dir data/toy
```

For a real scene, follow [docs/dataset_format.md](docs/dataset_format.md), set
`data.type: directory` and `data.root`, and verify whether `v_flip` is needed.

Every training run writes its exact config, environment metadata, per-step CSV,
summary JSON, checkpoints, progress PNGs, and `renders/training_progress.gif`.
The animation shows ground truth, current prediction, error, and the evolving
neural texture projected to PCA RGB. Evaluation also writes paired images.

## Distributed scaling

The global batch is fixed by config and divided across ranks; invalid divisions
fail explicitly. Local examples:

```bash
torchrun --standalone --nproc_per_node=2 -m neural_texture.train \
  --config configs/scaling.yaml --output-dir outputs/scaling_2gpu
python scripts/plot_results.py outputs/scaling_1gpu outputs/scaling_2gpu \
  --output-dir outputs/plots
```

For the communication optimization, compare
`configs/accumulation_baseline.yaml` with `configs/optimized.yaml`. Both have
identical effective workloads; only the former synchronizes every microstep.

## Voxel texture showcase

`configs/showcase.yaml` trains the same neural-texture architecture on an
original block-game atlas with staggered brick, clustered leaves, grass/dirt,
and cobblestone tiles. Each view observes a different local UV crop, making the
training animation visually recognizable while retaining exact reproducibility.
It does not copy assets from Minecraft or another commercial game.

```bash
python -m neural_texture.train \
  --config configs/showcase.yaml --output-dir outputs/showcase
```

On USC CARC:

```bash
sbatch slurm/showcase_1gpu_carc.sbatch
```

The presentation artifact is written to
`outputs/showcase_*/renders/training_progress.gif`.

## USC-inspired high-resolution showcase

`configs/usc_showcase.yaml` uses a 1254x1254 original image atlas containing
four cardinal-and-gold subjects: readable USC lettering, a Trojan helmet,
Romanesque campus architecture, and a Southern California palm sunset. The
model learns a 1024x1024x16 neural texture from randomized local UV views, so
the progress GIF exposes both coarse reconstruction and fine-detail learning.
The artwork is an unofficial project illustration, not an official USC mark.

```bash
sbatch slurm/usc_showcase_1gpu_carc.sbatch
```

The source target is `assets/usc_inspired_atlas.png`; the training animation is
written to `outputs/usc_showcase_1gpu_*/renders/training_progress.gif`.

For a separate strong-scaling measurement of this complex target, use
`configs/usc_scaling.yaml` unchanged on 1, 2, and 4 GPUs. It keeps the global
batch fixed at four and uses a shorter 300-step timing run; do not mix these
measurements with the simpler `configs/scaling.yaml` curve.

## USC CARC

CARC environment setup and 1-, 2-, and two-node 4-GPU Slurm templates are in
[docs/carc.md](docs/carc.md). The account is supplied at submission so no USC
project ID is committed. The full measurement contract is in
[docs/benchmarking.md](docs/benchmarking.md).

## Reproducibility and limitations

Seeds, resolved config, package/hardware metadata, and checkpoints are saved per
run. CUDA mixed precision and PyTorch Profiler are configurable. The synthetic
validation views use the same analytical signal but different UV offsets.
Current limitations include no SSIM/LPIPS and no sparse-tile synchronization.
The communication hook reports bucket payload and elapsed collective time; it
does not infer physical network traffic. No improvement is claimed without
CARC measurements.

