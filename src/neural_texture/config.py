"""Configuration loading and validation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 7,
    "device": "auto",
    "data": {
        "type": "synthetic",
        "root": None,
        "num_train_views": 4,
        "num_val_views": 2,
        "image_height": 32,
        "image_width": 32,
        "synthetic_pattern": "gradient",
        "synthetic_image": None,
        "view_scale": 1.0,
        "mask_shape": "full",
        "v_flip": False,
    },
    "model": {
        "texture_height": 32,
        "texture_width": 32,
        "texture_channels": 8,
        "renderer_width": 16,
        "renderer_depth": 3,
        "include_mask": True,
        "align_corners": True,
        "channels_last": False,
    },
    "training": {
        "batch_size": 2,
        "global_batch_size": None,
        "gradient_accumulation_steps": 1,
        "ddp_sync_each_microstep": False,
        "learning_rate": 0.01,
        "steps": 100,
        "warmup_steps": 5,
        "log_every": 10,
        "checkpoint_every": 50,
        "render_every": 25,
        "visibility_every": 10,
        "num_workers": 0,
        "mixed_precision": False,
        "fused_optimizer": False,
        "synchronize_each_phase": True,
        "measure_communication": True,
        "cudnn_benchmark": False,
        "resume": None,
    },
    "profiling": {
        "enabled": False,
        "wait_steps": 1,
        "warmup_steps": 1,
        "active_steps": 3,
        "record_shapes": False,
    },
}


def _merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML config and fill omitted values with safe defaults."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("The top-level YAML value must be a mapping.")
    config = _merge(DEFAULT_CONFIG, raw)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    for section in ("data", "model", "training"):
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"Missing configuration section: {section}")
    positive = {
        "data.image_height": config["data"]["image_height"],
        "data.image_width": config["data"]["image_width"],
        "model.texture_height": config["model"]["texture_height"],
        "model.texture_width": config["model"]["texture_width"],
        "model.texture_channels": config["model"]["texture_channels"],
        "training.batch_size": config["training"]["batch_size"],
        "training.steps": config["training"]["steps"],
        "training.gradient_accumulation_steps": config["training"]["gradient_accumulation_steps"],
    }
    for name, value in positive.items():
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if config["data"]["type"] not in {"synthetic", "directory"}:
        raise ValueError("data.type must be 'synthetic' or 'directory'.")
    if config["data"]["synthetic_pattern"] not in {
        "gradient", "neon_mosaic", "voxel_atlas", "image_atlas"
    }:
        raise ValueError(
            "data.synthetic_pattern must be 'gradient', 'neon_mosaic', 'voxel_atlas', "
            "or 'image_atlas'."
        )
    if config["data"]["synthetic_pattern"] == "image_atlas":
        image_path = config["data"].get("synthetic_image")
        if not isinstance(image_path, str) or not Path(image_path).is_file():
            raise ValueError("data.synthetic_image must name an existing image_atlas file.")
    view_scale = config["data"]["view_scale"]
    if not isinstance(view_scale, (int, float)) or not 0 < view_scale <= 1:
        raise ValueError("data.view_scale must be in (0, 1].")
    if config["data"]["mask_shape"] not in {"full", "ellipse"}:
        raise ValueError("data.mask_shape must be 'full' or 'ellipse'.")
    global_batch = config["training"].get("global_batch_size")
    if global_batch is not None and (not isinstance(global_batch, int) or global_batch <= 0):
        raise ValueError("training.global_batch_size must be null or a positive integer.")
    if not isinstance(config["training"]["ddp_sync_each_microstep"], bool):
        raise ValueError("training.ddp_sync_each_microstep must be boolean.")
    for key in (
        "fused_optimizer", "synchronize_each_phase", "measure_communication", "cudnn_benchmark"
    ):
        if not isinstance(config["training"][key], bool):
            raise ValueError(f"training.{key} must be boolean.")
    if not isinstance(config["model"]["channels_last"], bool):
        raise ValueError("model.channels_last must be boolean.")
    if config["training"]["measure_communication"] and not config["training"]["synchronize_each_phase"]:
        raise ValueError(
            "training.measure_communication requires synchronize_each_phase for accurate timing."
        )


def save_config(config: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

