import torch

from scripts.render_texture_target import render_voxel_atlas


def test_exact_voxel_target_has_no_clamped_top_rows_or_wrapped_edge():
    target = render_voxel_atlas(128)
    assert target.shape == (3, 128, 128)
    assert torch.all((target >= 0) & (target <= 1))
    # The top and second logical cell rows must differ; a clamped UV preview
    # previously repeated the top edge across many pixels.
    assert not torch.equal(target[:, 0], target[:, 16])
    assert not torch.equal(target[:, :, 0], target[:, :, -1])
