from copy import deepcopy

import pytest
import torch

from neural_texture.config import DEFAULT_CONFIG
from neural_texture.distributed import DistributedContext
from neural_texture.train import _batch_plan


def _context(world_size: int) -> DistributedContext:
    return DistributedContext(0, 0, world_size, torch.device("cpu"), "gloo")


def test_global_batch_is_divided_without_changing_workload():
    config = deepcopy(DEFAULT_CONFIG)
    config["training"]["global_batch_size"] = 16
    config["training"]["gradient_accumulation_steps"] = 2
    assert _batch_plan(config, _context(4)) == (16, 2, 2)


def test_invalid_global_batch_fails_instead_of_rounding():
    config = deepcopy(DEFAULT_CONFIG)
    config["training"]["global_batch_size"] = 10
    with pytest.raises(ValueError, match="must be divisible"):
        _batch_plan(config, _context(4))
