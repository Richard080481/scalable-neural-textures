import numpy as np
import pytest
import torch
from PIL import Image

from neural_texture.data import NeuralTextureDataset, SyntheticNeuralTextureDataset


def test_synthetic_dataset_shapes_and_ranges():
    sample = SyntheticNeuralTextureDataset(num_views=2, height=9, width=11)[0]
    assert sample["rgb"].shape == (3, 9, 11)
    assert sample["uv"].shape == (9, 11, 2)
    assert sample["mask"].shape == (1, 9, 11)
    assert torch.all((sample["uv"] >= 0) & (sample["uv"] <= 1))


def test_directory_dataset_loads_and_can_flip_v(tmp_path):
    for folder in ("rgb", "uv", "mask"):
        (tmp_path / "train" / folder).mkdir(parents=True)
    Image.fromarray(np.zeros((2, 3, 3), dtype=np.uint8)).save(tmp_path / "train/rgb/0000.png")
    Image.fromarray(np.full((2, 3), 255, dtype=np.uint8)).save(tmp_path / "train/mask/0000.png")
    uv = np.zeros((2, 3, 2), dtype=np.float32)
    uv[..., 1] = 0.25
    np.save(tmp_path / "train/uv/0000.npy", uv)
    sample = NeuralTextureDataset(tmp_path, v_flip=True)[0]
    assert torch.allclose(sample["uv"][..., 1], torch.full((2, 3), 0.75))


def test_directory_dataset_reports_missing_match(tmp_path):
    (tmp_path / "train/rgb").mkdir(parents=True)
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(tmp_path / "train/rgb/0000.png")
    with pytest.raises(FileNotFoundError, match="0000"):
        NeuralTextureDataset(tmp_path)

