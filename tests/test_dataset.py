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


def test_neon_mosaic_partial_view_and_ellipse_mask():
    sample = SyntheticNeuralTextureDataset(
        num_views=2,
        height=32,
        width=40,
        seed=11,
        pattern="neon_mosaic",
        view_scale=0.6,
        mask_shape="ellipse",
    )[0]
    assert sample["rgb"].shape == (3, 32, 40)
    assert float(sample["uv"][..., 0].max() - sample["uv"][..., 0].min()) == pytest.approx(0.6)
    assert sample["mask"].min() == 0
    assert sample["mask"].max() == 1
    assert torch.all((sample["rgb"] >= 0) & (sample["rgb"] <= 1))


def test_voxel_atlas_is_bounded_and_spatially_varied():
    sample = SyntheticNeuralTextureDataset(
        num_views=1,
        height=64,
        width=64,
        seed=4,
        pattern="voxel_atlas",
        view_scale=1.0,
    )[0]
    assert torch.all((sample["rgb"] >= 0) & (sample["rgb"] <= 1))
    assert float(sample["rgb"].std()) > 0.12


def test_explicit_visual_offset_has_exact_uv_extent():
    dataset = SyntheticNeuralTextureDataset(
        num_views=1, height=16, width=16, pattern="voxel_atlas", view_scale=0.48
    )
    sample = dataset.sample_at_offset((0.51, 0.01), "leaves")
    assert float(sample["uv"][..., 0].min()) == pytest.approx(0.51)
    assert float(sample["uv"][..., 0].max()) == pytest.approx(0.99)
    assert float(sample["uv"][..., 1].min()) == pytest.approx(0.01)
    assert float(sample["uv"][..., 1].max()) == pytest.approx(0.49)


def test_fixed_offsets_are_part_of_training_dataset():
    fixed = [[0.01, 0.01], [0.51, 0.51]]
    dataset = SyntheticNeuralTextureDataset(
        num_views=4,
        height=8,
        width=8,
        pattern="voxel_atlas",
        view_scale=0.48,
        fixed_offsets=fixed,
    )
    assert torch.allclose(dataset.offsets[:2], torch.tensor(fixed))
    first = dataset[0]
    assert float(first["uv"][..., 0].min()) == pytest.approx(0.01)
    assert float(first["uv"][..., 1].min()) == pytest.approx(0.01)


def test_image_atlas_samples_project_image(tmp_path):
    source = np.zeros((8, 8, 3), dtype=np.uint8)
    source[..., 0] = np.arange(8, dtype=np.uint8)[None, :] * 32
    source[..., 1] = np.arange(8, dtype=np.uint8)[:, None] * 32
    path = tmp_path / "atlas.png"
    Image.fromarray(source).save(path)
    sample = SyntheticNeuralTextureDataset(
        num_views=1,
        height=8,
        width=8,
        pattern="image_atlas",
        image_path=str(path),
        view_scale=0.5,
    )[0]
    assert sample["rgb"].shape == (3, 8, 8)
    assert float(sample["rgb"].std()) > 0.05
    assert torch.all((sample["rgb"] >= 0) & (sample["rgb"] <= 1))


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

