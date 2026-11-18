"""Masked image reconstruction metrics."""

from __future__ import annotations

import torch


def _denominator(mask: torch.Tensor, channels: int) -> torch.Tensor:
    return (mask.sum() * channels).clamp_min(1.0)


def masked_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return ((prediction - target).abs() * mask).sum() / _denominator(mask, prediction.shape[1])


def masked_psnr(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mse = (((prediction - target) ** 2) * mask).sum() / _denominator(mask, prediction.shape[1])
    return -10.0 * torch.log10(mse.clamp_min(1e-12))

