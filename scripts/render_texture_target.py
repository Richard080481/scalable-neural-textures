"""Render an exact, zero-offset analytical target texture at any resolution."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from neural_texture.data.synthetic import SyntheticNeuralTextureDataset


def render_voxel_atlas(resolution: int) -> torch.Tensor:
    if resolution < 2:
        raise ValueError("resolution must be at least two")
    coordinate = torch.arange(resolution, dtype=torch.float32) / resolution
    v, u = torch.meshgrid(coordinate, coordinate, indexing="ij")
    return SyntheticNeuralTextureDataset._voxel_atlas(u, v)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--output", type=Path, default=Path("outputs/voxel_target_1024.png"))
    args = parser.parse_args()
    texture = render_voxel_atlas(args.resolution)
    array = texture.permute(1, 2, 0).mul(255).round().byte().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array), mode="RGB").save(args.output)
    means = texture.mean(dim=(1, 2))
    print(
        f"saved={args.output} shape={tuple(texture.shape)} "
        f"min={float(texture.min()):.6f} max={float(texture.max()):.6f} "
        f"mean_rgb={[round(float(value), 6) for value in means]}"
    )


if __name__ == "__main__":
    main()
