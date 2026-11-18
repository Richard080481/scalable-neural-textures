# Benchmark protocol

## Strong scaling

Use `configs/scaling.yaml` unchanged for every GPU count. It fixes the global
batch at eight views and the optimizer workload at 500 steps. The trainer
derives the per-rank micro-batch as:

```text
micro_batch = global_batch / (world_size * accumulation_steps)
```

It exits rather than rounding when the division is not exact. Report timing
means after the configured warm-up steps. Run at least three trials on the same
GPU model, precision, and node topology; never combine unlike GPU models in one
scaling curve.

The DDP communication timer is installed as an averaged all-reduce hook. Its
duration is the exposed communication window from the first asynchronous bucket
launch until a device synchronization after backward confirms completion.
`communication_bytes` counts bucket payload per rank; it is not a claim about
topology-dependent wire traffic. Backward time includes this window, so
communication time is a diagnostic component and must not be added to the
other components as if it were disjoint.

## Visibility

At `visibility_every`, valid UV samples are quantized to their nearest texture
texel. The run records per-rank visible ratio, the all-rank union, and the
fraction of globally visible texels touched by more than one rank. This nearest
texel statistic is an interpretable proxy; bilinear sampling actually touches
up to four neighboring texels.

## Communication optimization

Compare `configs/accumulation_baseline.yaml` and `configs/optimized.yaml` using
the same GPU count. Both use the same global batch, four microsteps, optimizer
steps, seed, model, and samples. The baseline all-reduces every microstep; the
optimized config uses DDP `no_sync()` for the first three microsteps. Compare
communication bytes/time, throughput, final PSNR, and renders.

## Visual evidence

Every run produces `renders/training_progress.gif`. Each frame shows ground
truth, prediction, absolute error, and a PCA projection of all neural-texture
feature channels. `scripts/plot_results.py` creates loss, PSNR, timing,
visibility, speedup, efficiency, and throughput plots strictly from saved
CSV/JSON results.

