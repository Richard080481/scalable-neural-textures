import torch

from neural_texture.distributed import DistributedContext
from neural_texture.metrics import distributed_visibility, visible_texel_mask


def test_visible_texels_follow_uv_and_mask():
    uv = torch.tensor([[[[0.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [1.0, 1.0]]]])
    mask = torch.tensor([[[[1.0, 1.0], [0.0, 1.0]]]])
    visible = visible_texel_mask(uv, mask, texture_height=2, texture_width=2)
    assert torch.equal(visible, torch.tensor([True, True, False, True]))


def test_single_rank_visibility_statistics():
    visible = torch.tensor([True, False, True, False])
    context = DistributedContext(rank=0, local_rank=0, world_size=1, device=torch.device("cpu"), backend=None)
    stats = distributed_visibility(visible, context)
    assert stats == {
        "visible_texel_ratio_local": 0.5,
        "visible_texel_ratio_global": 0.5,
        "visible_texel_overlap_ratio": 1.0,
    }
