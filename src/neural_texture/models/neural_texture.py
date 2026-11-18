"""Learnable feature texture and differentiable UV sampling."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class NeuralTexture(nn.Module):
    """A learnable [1, C, H, W] texture sampled by UV coordinates in [0, 1]."""

    def __init__(self, channels: int, height: int, width: int, align_corners: bool = True):
        super().__init__()
        self.align_corners = align_corners
        self.texture = nn.Parameter(torch.empty(1, channels, height, width))
        nn.init.normal_(self.texture, mean=0.0, std=0.01)

    def forward(self, uv: torch.Tensor) -> torch.Tensor:
        if uv.ndim != 4 or uv.shape[-1] != 2:
            raise ValueError(f"UV must have shape [B, H, W, 2], got {tuple(uv.shape)}")
        # grid_sample uses (x, y), so UV maps directly to (horizontal u, vertical v).
        grid = uv.mul(2.0).sub(1.0)
        texture = self.texture.expand(uv.shape[0], -1, -1, -1)
        return F.grid_sample(
            texture,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=self.align_corners,
        )

