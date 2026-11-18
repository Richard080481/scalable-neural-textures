"""Combined texture sampler and renderer."""

from __future__ import annotations

import torch
from torch import nn

from .neural_texture import NeuralTexture
from .renderer import NeuralRenderer


class NeuralTextureModel(nn.Module):
    def __init__(
        self,
        texture_channels: int,
        texture_height: int,
        texture_width: int,
        renderer_width: int = 32,
        renderer_depth: int = 3,
        include_mask: bool = True,
        align_corners: bool = True,
    ):
        super().__init__()
        self.include_mask = include_mask
        self.neural_texture = NeuralTexture(texture_channels, texture_height, texture_width, align_corners)
        self.renderer = NeuralRenderer(texture_channels + int(include_mask), renderer_width, renderer_depth)

    @classmethod
    def from_config(cls, config: dict) -> "NeuralTextureModel":
        model = config["model"]
        return cls(**model)

    def forward(self, uv: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        features = self.neural_texture(uv)
        if self.include_mask:
            if mask is None:
                raise ValueError("A mask is required when include_mask is enabled.")
            features = torch.cat((features, mask), dim=1)
        return self.renderer(features)

