"""Build the README result figure from the committed aggregate CARC metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def selected(
    rows: list[dict[str, str]], experiment: str, variant: str
) -> list[dict[str, str]]:
    result = [
        row for row in rows
        if row["experiment"] == experiment and row["variant"] == variant
    ]
    return sorted(result, key=lambda row: int(row["gpus"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=Path("results/final_metrics.csv"))
    parser.add_argument("--no-sync", type=Path, default=Path("results/no_sync_metrics.csv"))
    parser.add_argument(
        "--output", type=Path, default=Path("docs/assets/results_overview.png")
    )
    args = parser.parse_args()
    rows = read_rows(args.metrics)
    sync_rows = read_rows(args.no_sync)

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    figure.suptitle("USC CARC A40 results (mean ± std, 3 trials)", fontsize=15, weight="bold")

    # A: workload size decides whether strong scaling is positive.
    for experiment, label, color in (
        ("complex-small", "Global batch 4", "#990000"),
        ("complex-large", "Global batch 64", "#ffcc00"),
    ):
        group = selected(rows, experiment, "optimized")
        gpus = [int(row["gpus"]) for row in group]
        times = [float(row["mean_step_ms"]) for row in group]
        speedup = [times[0] / value for value in times]
        axes[0].plot(gpus, speedup, "o-", linewidth=2.5, markersize=7, label=label, color=color)
    axes[0].plot([1, 2, 4], [1, 2, 4], "--", color="#777777", label="Ideal")
    axes[0].set(title="A. Strong-scaling speedup", xlabel="A40 GPUs", ylabel="Speedup", xticks=[1, 2, 4])
    axes[0].legend(frameon=True, fontsize=9)

    # B: execution changes improve absolute speed at every GPU count.
    baseline = selected(rows, "complex-small", "baseline")
    optimized = selected(rows, "complex-small", "optimized")
    x = list(range(3))
    width = 0.36
    axes[1].bar(
        [value - width / 2 for value in x],
        [float(row["mean_step_ms"]) for row in baseline],
        width,
        yerr=[float(row["std_step_ms"]) for row in baseline],
        capsize=4,
        color="#777777",
        label="Instrumented baseline",
    )
    axes[1].bar(
        [value + width / 2 for value in x],
        [float(row["mean_step_ms"]) for row in optimized],
        width,
        yerr=[float(row["std_step_ms"]) for row in optimized],
        capsize=4,
        color="#990000",
        label="Optimized",
    )
    axes[1].set(
        title="B. Optimized execution",
        xlabel="A40 GPUs",
        ylabel="Step time (ms, lower is better)",
        xticks=x,
        xticklabels=["1", "2", "4"],
    )
    axes[1].legend(frameon=True, fontsize=9)

    # C: no_sync communicates one quarter of the gradient payload.
    labels = ["Sync every\nmicrostep", "DDP no_sync()"]
    payload = [float(row["comm_bytes"]) / 1e6 for row in sync_rows]
    colors = ["#777777", "#990000"]
    bars = axes[2].bar(labels, payload, color=colors, width=0.58)
    axes[2].bar_label(bars, fmt="%.1f MB", padding=3)
    axes[2].set(title="C. Communication payload", ylabel="MB per rank / optimizer step", ylim=(0, 78))
    axes[2].text(
        0.5, 0.89, "75% less payload\n4.5% faster steps",
        transform=axes[2].transAxes, ha="center", va="top", fontsize=11, weight="bold",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#fff4bf", "edgecolor": "#c8a600"},
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, facecolor="white")
    plt.close(figure)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
