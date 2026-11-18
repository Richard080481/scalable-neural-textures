"""Deterministic analytical dataset used by tests and smoke runs."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset


class SyntheticNeuralTextureDataset(Dataset[dict[str, Any]]):
    """Create deterministic views of an analytical UV-defined appearance."""

    def __init__(
        self,
        num_views: int = 4,
        height: int = 32,
        width: int = 32,
        seed: int = 7,
        pattern: str = "gradient",
        view_scale: float = 1.0,
        mask_shape: str = "full",
    ):
        if num_views <= 0 or height <= 1 or width <= 1:
            raise ValueError("num_views must be positive and spatial sizes must exceed one.")
        if pattern not in {"gradient", "neon_mosaic", "voxel_atlas"}:
            raise ValueError(f"Unknown synthetic pattern: {pattern}")
        if not 0 < view_scale <= 1:
            raise ValueError("view_scale must be in (0, 1].")
        if mask_shape not in {"full", "ellipse"}:
            raise ValueError(f"Unknown mask shape: {mask_shape}")
        self.num_views = num_views
        self.height = height
        self.width = width
        self.pattern = pattern
        self.view_scale = view_scale
        self.mask_shape = mask_shape
        generator = torch.Generator().manual_seed(seed)
        if view_scale < 1.0:
            self.offsets = torch.rand(num_views, 2, generator=generator) * (1.0 - view_scale)
        else:
            self.offsets = (torch.rand(num_views, 2, generator=generator) - 0.5) * 0.16

    def __len__(self) -> int:
        return self.num_views

    def __getitem__(self, index: int) -> dict[str, Any]:
        y = torch.linspace(0.0, 1.0, self.height)
        x = torch.linspace(0.0, 1.0, self.width)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        uv = torch.stack((xx, yy), dim=-1)
        uv = (uv * self.view_scale + self.offsets[index]).clamp(0.0, 1.0)
        u, v = uv[..., 0], uv[..., 1]
        if self.pattern == "gradient":
            rgb = torch.stack((
                u,
                v,
                0.5 + 0.25 * torch.sin(2.0 * torch.pi * u) * torch.cos(2.0 * torch.pi * v),
            ))
        elif self.pattern == "neon_mosaic":
            rgb = self._neon_mosaic(u, v)
        else:
            rgb = self._voxel_atlas(u, v)
        if self.mask_shape == "ellipse":
            ellipse = ((xx - 0.5) / 0.49) ** 2 + ((yy - 0.5) / 0.43) ** 2 <= 1.0
            mask = ellipse.float().unsqueeze(0)
            rgb = rgb * mask
        else:
            mask = torch.ones(1, self.height, self.width)
        return {"rgb": rgb.float(), "uv": uv.float(), "mask": mask, "name": f"synthetic_{index:04d}"}

    @staticmethod
    def _neon_mosaic(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """A colorful multi-scale pattern with rings, weave, glow, and angular bands."""
        x, y = u - 0.5, v - 0.5
        radius = torch.sqrt(x.square() + y.square() + 1e-8)
        angle = torch.atan2(y, x)
        rings = 0.5 + 0.5 * torch.cos(18.0 * torch.pi * radius - 2.5 * torch.sin(5.0 * angle))
        weave = 0.5 + 0.5 * torch.sin(16.0 * torch.pi * u) * torch.sin(16.0 * torch.pi * v)
        angular = 0.5 + 0.5 * torch.sin(7.0 * angle + 12.0 * radius)
        glow_a = torch.exp(-45.0 * ((u - 0.28).square() + (v - 0.32).square()))
        glow_b = torch.exp(-55.0 * ((u - 0.72).square() + (v - 0.68).square()))
        fine_lines = (0.5 + 0.5 * torch.cos(32.0 * torch.pi * (u + 0.35 * v))).pow(8)
        red = 0.04 + 0.58 * rings + 0.28 * weave + 0.72 * glow_a
        green = 0.03 + 0.48 * (1.0 - rings) + 0.34 * angular + 0.68 * glow_b
        blue = 0.10 + 0.48 * angular + 0.35 * weave + 0.30 * fine_lines
        vignette = (1.15 - 0.72 * radius).clamp(0.55, 1.0)
        return (torch.stack((red, green, blue)) * vignette).clamp(0.0, 1.0)

    @staticmethod
    def _voxel_atlas(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """An original block-game atlas: brick, leaves, grass/dirt, and stone."""
        local_u = torch.remainder(u * 2.0, 1.0)
        local_v = torch.remainder(v * 2.0, 1.0)
        cell_x = torch.floor(local_u * 16.0)
        cell_y = torch.floor(local_v * 16.0)
        noise = torch.frac(torch.sin(cell_x * 12.9898 + cell_y * 78.233) * 43758.5453).abs()

        # Staggered terracotta bricks with pale mortar and per-brick variation.
        brick_row = torch.floor(local_v * 8.0)
        shifted_u = torch.remainder(local_u + torch.remainder(brick_row, 2.0) * 0.125, 1.0)
        mortar = (torch.remainder(shifted_u * 4.0, 1.0) < 0.07) | (
            torch.remainder(local_v * 8.0, 1.0) < 0.10
        )
        brick_variation = 0.76 + 0.24 * noise
        brick = torch.stack((
            0.68 * brick_variation,
            0.24 * brick_variation,
            0.12 * brick_variation,
        ))
        mortar_color = torch.tensor([0.76, 0.67, 0.54], device=u.device)[:, None, None]
        brick = torch.where(mortar.unsqueeze(0), mortar_color, brick)

        # Layered leaf clusters with rare dark gaps and lime highlights.
        leaf_cluster = 0.35 + 0.65 * noise
        leaf = torch.stack((
            0.05 + 0.12 * leaf_cluster,
            0.24 + 0.58 * leaf_cluster,
            0.08 + 0.20 * leaf_cluster,
        ))
        leaf = torch.where((noise < 0.075).unsqueeze(0), leaf * 0.16, leaf)
        leaf = torch.where((noise > 0.90).unsqueeze(0), (leaf + 0.20).clamp(0, 1), leaf)

        # A grass block side: green cap and hanging roots above noisy soil.
        soil = torch.stack((
            0.34 + 0.18 * noise,
            0.19 + 0.12 * noise,
            0.08 + 0.07 * noise,
        ))
        grass_edge = 0.25 + 0.10 * torch.sin(cell_x * 2.4 + noise * 5.0)
        grassy = local_v < grass_edge
        grass = torch.stack((
            0.12 + 0.15 * noise,
            0.42 + 0.40 * noise,
            0.08 + 0.12 * noise,
        ))
        grass_block = torch.where(grassy.unsqueeze(0), grass, soil)
        root_fleck = (~grassy) & (noise > 0.94)
        grass_block = torch.where(root_fleck.unsqueeze(0), grass * 0.72, grass_block)

        # Cool cobblestone cells with dark seams and irregular highlights.
        seam = (torch.remainder(local_u * 8.0, 1.0) < 0.08) | (
            torch.remainder(local_v * 8.0, 1.0) < 0.08
        )
        stone_value = 0.30 + 0.36 * noise
        stone = torch.stack((stone_value * 0.88, stone_value * 0.96, stone_value))
        stone = torch.where(seam.unsqueeze(0), stone * 0.38, stone)

        top, left = v < 0.5, u < 0.5
        rgb = torch.where((top & left).unsqueeze(0), brick, leaf)
        rgb = torch.where(((~top) & left).unsqueeze(0), grass_block, rgb)
        rgb = torch.where(((~top) & (~left)).unsqueeze(0), stone, rgb)
        return rgb.clamp(0.0, 1.0)

