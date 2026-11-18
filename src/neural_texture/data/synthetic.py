"""Deterministic analytical dataset used by tests and smoke runs."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset


class SyntheticNeuralTextureDataset(Dataset[dict[str, Any]]):
    """Create views of a smooth UV-defined RGB signal with small UV offsets."""

    def __init__(self, num_views: int = 4, height: int = 32, width: int = 32, seed: int = 7):
        if num_views <= 0 or height <= 1 or width <= 1:
            raise ValueError("num_views must be positive and spatial sizes must exceed one.")
        self.num_views = num_views
        self.height = height
        self.width = width
        generator = torch.Generator().manual_seed(seed)
        self.offsets = (torch.rand(num_views, 2, generator=generator) - 0.5) * 0.16

    def __len__(self) -> int:
        return self.num_views

    def __getitem__(self, index: int) -> dict[str, Any]:
        y = torch.linspace(0.0, 1.0, self.height)
        x = torch.linspace(0.0, 1.0, self.width)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        uv = torch.stack((xx, yy), dim=-1)
        uv = (uv + self.offsets[index]).clamp(0.0, 1.0)
        u, v = uv[..., 0], uv[..., 1]
        rgb = torch.stack((u, v, 0.5 + 0.25 * torch.sin(2.0 * torch.pi * u) * torch.cos(2.0 * torch.pi * v)))
        mask = torch.ones(1, self.height, self.width)
        return {"rgb": rgb.float(), "uv": uv.float(), "mask": mask, "name": f"synthetic_{index:04d}"}

