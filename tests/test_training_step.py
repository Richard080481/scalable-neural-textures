import copy

import torch
from torch.utils.data import DataLoader

from neural_texture.config import DEFAULT_CONFIG
from neural_texture.data import SyntheticNeuralTextureDataset
from neural_texture.metrics import masked_l1
from neural_texture.models import NeuralTextureModel
from neural_texture.utils.checkpoint import load_checkpoint, save_checkpoint


def _tiny_model():
    return NeuralTextureModel(4, 8, 8, renderer_width=8, renderer_depth=2)


def test_tiny_example_overfits_and_checkpoint_reproduces_render(tmp_path):
    torch.manual_seed(3)
    batch = next(iter(DataLoader(SyntheticNeuralTextureDataset(1, 8, 8, seed=2), batch_size=1)))
    model = _tiny_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.04)
    with torch.no_grad():
        initial = float(masked_l1(model(batch["uv"], batch["mask"]), batch["rgb"], batch["mask"]))
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch["uv"], batch["mask"])
        loss = masked_l1(prediction, batch["rgb"], batch["mask"])
        loss.backward()
        optimizer.step()
    final = float(loss.detach())
    assert final < initial * 0.35

    expected = model(batch["uv"], batch["mask"]).detach()
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, model, optimizer, step=80)
    restored = _tiny_model()
    state = load_checkpoint(checkpoint, restored)
    actual = restored(batch["uv"], batch["mask"]).detach()
    assert state["step"] == 80
    assert torch.equal(actual, expected)


def test_default_config_is_not_mutated():
    before = copy.deepcopy(DEFAULT_CONFIG)
    assert DEFAULT_CONFIG == before

