"""Materialize the analytical toy dataset in the documented on-disk format."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from neural_texture.data import SyntheticNeuralTextureDataset


def write_split(
    root: Path,
    split: str,
    views: int,
    height: int,
    width: int,
    seed: int,
    pattern: str,
    view_scale: float,
    mask_shape: str,
    image_path: str | None,
) -> None:
    target = root / split
    for folder in ("rgb", "uv", "mask"):
        (target / folder).mkdir(parents=True, exist_ok=True)
    dataset = SyntheticNeuralTextureDataset(
        views, height, width, seed,
        pattern=pattern, view_scale=view_scale, mask_shape=mask_shape, image_path=image_path,
    )
    for index, sample in enumerate(dataset):
        name = f"{index:04d}"
        rgb = sample["rgb"].permute(1, 2, 0).mul(255).round().byte().numpy()
        mask = sample["mask"][0].mul(255).byte().numpy()
        Image.fromarray(rgb).save(target / "rgb" / f"{name}.png")
        Image.fromarray(mask).save(target / "mask" / f"{name}.png")
        np.save(target / "uv" / f"{name}.npy", sample["uv"].numpy())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-views", type=int, default=8)
    parser.add_argument("--val-views", type=int, default=2)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--pattern",
        choices=("gradient", "neon_mosaic", "voxel_atlas", "image_atlas"),
        default="gradient",
    )
    parser.add_argument("--image-path")
    parser.add_argument("--view-scale", type=float, default=1.0)
    parser.add_argument("--mask-shape", choices=("full", "ellipse"), default="full")
    args = parser.parse_args()
    write_split(
        args.output_dir, "train", args.train_views, args.height, args.width, args.seed,
        args.pattern, args.view_scale, args.mask_shape, args.image_path,
    )
    write_split(
        args.output_dir, "val", args.val_views, args.height, args.width, args.seed + 10_000,
        args.pattern, args.view_scale, args.mask_shape, args.image_path,
    )


if __name__ == "__main__":
    main()

