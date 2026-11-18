"""Create publication-ready training and scaling plots from run outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_metrics(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append({key: float(value) for key, value in row.items() if value and key != "device"})
        return rows


def plot_training(run_dir: Path, output_dir: Path) -> None:
    rows = _read_metrics(run_dir / "metrics.csv")
    steps = [row["step"] for row in rows]
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    axes[0, 0].plot(steps, [row["loss"] for row in rows])
    axes[0, 0].set(title="Masked L1 loss", xlabel="Optimizer step", ylabel="Loss")
    axes[0, 1].plot(steps, [row["psnr"] for row in rows], color="tab:green")
    axes[0, 1].set(title="Rendering quality", xlabel="Optimizer step", ylabel="PSNR (dB)")
    for key, label in (
        ("data_time_ms", "Data"), ("forward_time_ms", "Forward"),
        ("backward_time_ms", "Backward + DDP"), ("communication_time_ms", "All-reduce"),
        ("optimizer_time_ms", "Optimizer"),
    ):
        axes[1, 0].plot(steps, [row[key] for row in rows], label=label, alpha=0.85)
    axes[1, 0].set(title="Step-time components", xlabel="Optimizer step", ylabel="Milliseconds")
    axes[1, 0].legend(fontsize=8)
    visible = [row for row in rows if "visible_texel_ratio_global" in row]
    axes[1, 1].plot(
        [row["step"] for row in visible], [100 * row["visible_texel_ratio_global"] for row in visible],
        marker="o", label="Global visible",
    )
    axes[1, 1].plot(
        [row["step"] for row in visible], [100 * row["visible_texel_overlap_ratio"] for row in visible],
        marker="s", label="Cross-rank overlap",
    )
    axes[1, 1].set(title="UV visibility", xlabel="Optimizer step", ylabel="Percent", ylim=(0, 105))
    axes[1, 1].legend(fontsize=8)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / f"{run_dir.name}_training.png", dpi=180)
    plt.close(figure)


def plot_scaling(run_dirs: list[Path], output_dir: Path) -> None:
    summaries = [_read_json(path / "summary.json") | {"run": path.name} for path in run_dirs]
    summaries.sort(key=lambda item: item["world_size"])
    baseline = next((item for item in summaries if item["world_size"] == 1), None)
    if baseline is None:
        return
    worlds = [item["world_size"] for item in summaries]
    speedups = [baseline["mean_step_time_ms"] / item["mean_step_time_ms"] for item in summaries]
    efficiencies = [100 * speedup / world for speedup, world in zip(speedups, worlds)]
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    axes[0].plot(worlds, speedups, "o-", label="Measured")
    axes[0].plot(worlds, worlds, "--", color="gray", label="Ideal")
    axes[0].set(title="Strong-scaling speedup", xlabel="GPUs", ylabel="Speedup", xticks=worlds)
    axes[0].legend()
    axes[1].plot(worlds, efficiencies, "o-", color="tab:orange")
    axes[1].axhline(100, linestyle="--", color="gray")
    axes[1].set(title="Parallel efficiency", xlabel="GPUs", ylabel="Efficiency (%)", xticks=worlds)
    axes[2].plot(worlds, [item["mean_samples_per_second"] for item in summaries], "o-", label="Throughput")
    axes[2].set(title="Throughput", xlabel="GPUs", ylabel="Samples / second", xticks=worlds)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "strong_scaling.png", dpi=180)
    plt.close(figure)
    with (output_dir / "scaling_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "run", "world_size", "mean_step_time_ms", "mean_samples_per_second",
            "mean_communication_time_ms", "speedup", "parallel_efficiency_percent",
            "final_psnr", "validation_psnr", "texture_gradient_bytes_fp32",
        ])
        writer.writeheader()
        for item, speedup, efficiency in zip(summaries, speedups, efficiencies):
            writer.writerow({
                **{key: item[key] for key in writer.fieldnames if key in item},
                "speedup": speedup,
                "parallel_efficiency_percent": efficiency,
            })


def plot_optimization(run_dirs: list[Path], output_dir: Path) -> None:
    summaries = [_read_json(path / "summary.json") | {"run": path.name} for path in run_dirs]
    baseline = next((item for item in summaries if item.get("ddp_sync_each_microstep") is True), None)
    optimized = next((item for item in summaries if item.get("ddp_sync_each_microstep") is False and item.get("gradient_accumulation_steps", 1) > 1), None)
    if baseline is None or optimized is None or baseline["world_size"] != optimized["world_size"]:
        return
    labels = ["Sync every\nmicrostep", "DDP no_sync()"]
    colors = ["tab:red", "tab:green"]
    figure, axes = plt.subplots(1, 4, figsize=(13, 3.6), constrained_layout=True)
    comparisons = (
        ("mean_communication_time_ms", "Communication time", "Milliseconds"),
        ("mean_communication_bytes", "Bucket payload / step", "Bytes per rank"),
        ("mean_samples_per_second", "Throughput", "Samples / second"),
        ("validation_psnr", "Validation quality", "PSNR (dB)"),
    )
    for axis, (key, title, ylabel) in zip(axes, comparisons):
        axis.bar(labels, [baseline[key], optimized[key]], color=colors)
        axis.set(title=title, ylabel=ylabel)
        axis.tick_params(axis="x", labelsize=8)
    figure.savefig(output_dir / "optimization_comparison.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/plots"))
    args = parser.parse_args()
    valid = [path for path in args.runs if (path / "summary.json").is_file()]
    if not valid:
        raise FileNotFoundError("No run directory containing summary.json was provided.")
    for run_dir in valid:
        plot_training(run_dir, args.output_dir)
    plot_scaling(valid, args.output_dir)
    plot_optimization(valid, args.output_dir)


if __name__ == "__main__":
    main()

