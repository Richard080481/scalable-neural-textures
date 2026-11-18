"""Rank-zero visual artifacts showing rendering and texture evolution."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F

from .models import NeuralTextureModel


def _rgb_image(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().float().clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(np.asarray(array), mode="RGB")


def texture_pca_image(texture: torch.Tensor, max_size: int = 256) -> Image.Image:
    """Project feature channels to RGB after bounded spatial downsampling."""
    features = texture.detach().float()
    height, width = features.shape[-2:]
    scale = min(1.0, max_size / max(height, width))
    out_h, out_w = max(1, round(height * scale)), max(1, round(width * scale))
    if (out_h, out_w) != (height, width):
        features = F.interpolate(features, size=(out_h, out_w), mode="bilinear", align_corners=False)
    channels = features.shape[1]
    samples = features[0].permute(1, 2, 0).reshape(-1, channels)
    centered = samples - samples.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(samples.shape[0] - 1, 1)
    _, eigenvectors = torch.linalg.eigh(covariance)
    basis = eigenvectors[:, -min(3, channels):].flip(1)
    # Remove PCA's arbitrary sign ambiguity so animation colors do not flip.
    dominant_rows = basis.abs().argmax(dim=0)
    signs = basis[dominant_rows, torch.arange(basis.shape[1], device=basis.device)].sign()
    basis = basis * torch.where(signs == 0, torch.ones_like(signs), signs)
    components = centered @ basis
    if components.shape[1] < 3:
        components = F.pad(components, (0, 3 - components.shape[1]))
    low = torch.quantile(components, 0.01, dim=0)
    high = torch.quantile(components, 0.99, dim=0)
    colors = ((components - low) / (high - low).clamp_min(1e-6)).clamp(0, 1)
    colors = colors.reshape(out_h, out_w, 3).mul(255).byte().cpu().numpy()
    return Image.fromarray(np.asarray(colors), mode="RGB")


@torch.no_grad()
def save_progress_frame(
    model: NeuralTextureModel,
    uv: torch.Tensor,
    mask: torch.Tensor,
    ground_truth: torch.Tensor,
    step: int,
    output_dir: Path,
) -> Path:
    """Save GT, prediction, error, and neural-texture PCA as one presentation frame."""
    was_training = model.training
    model.eval()
    prediction = model(uv[:1], mask[:1])[0]
    target = ground_truth[0]
    error = (prediction - target).abs()
    gt_image = _rgb_image(target)
    pred_image = _rgb_image(prediction)
    error_image = _rgb_image((error / 0.35).clamp(0, 1))
    display_scale = max(1.0, 160 / max(gt_image.size))
    display_size = (round(gt_image.width * display_scale), round(gt_image.height * display_scale))
    gt_image = gt_image.resize(display_size, Image.Resampling.NEAREST)
    pred_image = pred_image.resize(display_size, Image.Resampling.NEAREST)
    error_image = error_image.resize(display_size, Image.Resampling.NEAREST)
    pca_image = texture_pca_image(model.neural_texture.texture).resize(gt_image.size, Image.Resampling.BILINEAR)
    labels = ("Ground truth", "Prediction", "Absolute error", "Texture PCA")
    panels = (gt_image, pred_image, error_image, pca_image)
    header = 28
    canvas = Image.new("RGB", (gt_image.width * 4, gt_image.height + header), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, panel) in enumerate(zip(labels, panels)):
        x = index * gt_image.width
        canvas.paste(panel, (x, header))
        draw.text((x + 5, 7), label, fill="black")
    draw.text((canvas.width - 80, 7), f"step {step}", fill="black")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"progress_{step:06d}.png"
    canvas.save(path)
    if was_training:
        model.train()
    return path


def create_progress_gif(frame_dir: Path, output_path: Path, duration_ms: int = 350) -> None:
    frames = [Image.open(path).convert("RGB") for path in sorted(frame_dir.glob("progress_*.png"))]
    if not frames:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
    for frame in frames:
        frame.close()

