"""On-disk dataset interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .synthetic import SyntheticNeuralTextureDataset


def _image_tensor(path: Path, channels: int) -> torch.Tensor:
    mode = "RGB" if channels == 3 else "L"
    array = np.asarray(Image.open(path).convert(mode), dtype=np.float32) / 255.0
    if channels == 1:
        array = array[..., None]
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class NeuralTextureDataset(Dataset[dict[str, Any]]):
    """Load matching RGB, UV, and mask files from one split directory."""

    def __init__(self, root: str | Path, split: str = "train", v_flip: bool = False):
        self.split_dir = Path(root) / split
        self.v_flip = v_flip
        rgb_dir = self.split_dir / "rgb"
        if not rgb_dir.is_dir():
            raise FileNotFoundError(f"Missing RGB directory: {rgb_dir}")
        self.names = sorted(path.stem for path in rgb_dir.glob("*.png"))
        if not self.names:
            raise FileNotFoundError(f"No PNG images found in: {rgb_dir}")
        for name in self.names:
            for path in (
                rgb_dir / f"{name}.png",
                self.split_dir / "uv" / f"{name}.npy",
                self.split_dir / "mask" / f"{name}.png",
            ):
                if not path.is_file():
                    raise FileNotFoundError(f"Missing dataset file for view {name}: {path}")

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> dict[str, Any]:
        name = self.names[index]
        rgb = _image_tensor(self.split_dir / "rgb" / f"{name}.png", 3)
        mask = _image_tensor(self.split_dir / "mask" / f"{name}.png", 1)
        uv_array = np.load(self.split_dir / "uv" / f"{name}.npy").astype(np.float32)
        if uv_array.ndim != 3 or uv_array.shape[-1] != 2:
            raise ValueError(f"UV map {name} must have shape [H, W, 2], got {uv_array.shape}")
        uv = torch.from_numpy(uv_array)
        if self.v_flip:
            uv = uv.clone()
            uv[..., 1] = 1.0 - uv[..., 1]
        if uv.shape[:2] != rgb.shape[1:] or mask.shape[1:] != rgb.shape[1:]:
            raise ValueError(f"Spatial dimensions do not match for view {name}.")
        return {"rgb": rgb, "uv": uv, "mask": mask, "name": name}


def create_dataset(config: dict[str, Any], split: str) -> Dataset[dict[str, Any]]:
    data = config["data"]
    if data["type"] == "synthetic":
        count = data["num_train_views"] if split == "train" else data["num_val_views"]
        return SyntheticNeuralTextureDataset(
            num_views=count,
            height=data["image_height"],
            width=data["image_width"],
            seed=config["seed"] + (0 if split == "train" else 10_000),
            pattern=data["synthetic_pattern"],
            view_scale=float(data["view_scale"]),
            mask_shape=data["mask_shape"],
            image_path=data.get("synthetic_image"),
            fixed_offsets=data.get("train_fixed_offsets") if split == "train" else None,
        )
    if not data.get("root"):
        raise ValueError("data.root is required when data.type is 'directory'.")
    return NeuralTextureDataset(data["root"], split=split, v_flip=data["v_flip"])

