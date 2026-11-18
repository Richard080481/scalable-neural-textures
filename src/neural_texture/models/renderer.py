"""Compact convolutional neural renderer."""

from __future__ import annotations

from torch import nn


class NeuralRenderer(nn.Module):
    def __init__(self, input_channels: int, width: int = 32, depth: int = 3):
        super().__init__()
        if depth < 2:
            raise ValueError("Renderer depth must be at least two.")
        layers: list[nn.Module] = [nn.Conv2d(input_channels, width, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers.extend((nn.Conv2d(width, width, 3, padding=1), nn.ReLU(inplace=True)))
        layers.extend((nn.Conv2d(width, 3, 1), nn.Sigmoid()))
        self.network = nn.Sequential(*layers)

    def forward(self, features):
        return self.network(features)

