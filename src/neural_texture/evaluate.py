"""Evaluate a trained checkpoint and save rendered PNG images."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .config import load_config
from .data import create_dataset
from .metrics import masked_psnr
from .models import NeuralTextureModel
from .train import choose_device
from .utils.checkpoint import load_checkpoint
from .utils.logging import write_json


def _save_image(tensor: torch.Tensor, path: Path) -> None:
    array = tensor.detach().clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(np.asarray(array)).save(path)


@torch.no_grad()
def evaluate(config_path: Path, checkpoint: Path, output_dir: Path, split: str = "val") -> dict[str, float]:
    config = load_config(config_path)
    device = choose_device(config["device"])
    dataset = create_dataset(config, split)
    model = NeuralTextureModel.from_config(config).to(device)
    load_checkpoint(checkpoint, model, map_location=device)
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    values = []
    for batch in DataLoader(dataset, batch_size=1, shuffle=False):
        rgb, uv, mask = (batch[key].to(device) for key in ("rgb", "uv", "mask"))
        prediction = model(uv, mask)
        values.append(float(masked_psnr(prediction, rgb, mask)))
        _save_image(prediction[0], output_dir / f"{batch['name'][0]}_prediction.png")
        _save_image(rgb[0], output_dir / f"{batch['name'][0]}_ground_truth.png")
    result = {"psnr": sum(values) / len(values), "num_views": float(len(values))}
    write_json(output_dir / "evaluation.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    args = parser.parse_args()
    print(evaluate(args.config, args.checkpoint, args.output_dir, args.split))


if __name__ == "__main__":
    main()

