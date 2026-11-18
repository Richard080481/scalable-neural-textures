"""UV-to-texel visibility statistics for the communication study."""

from __future__ import annotations

import torch
import torch.distributed as dist

from neural_texture.distributed import DistributedContext


def visible_texel_mask(
    uv: torch.Tensor,
    valid_mask: torch.Tensor,
    texture_height: int,
    texture_width: int,
) -> torch.Tensor:
    """Return a flattened boolean mask of nearest texels touched by valid UVs."""
    valid = valid_mask[:, 0] > 0.5
    u = uv[..., 0].clamp(0, 1)
    v = uv[..., 1].clamp(0, 1)
    x = torch.round(u * (texture_width - 1)).long()
    y = torch.round(v * (texture_height - 1)).long()
    indices = (y * texture_width + x)[valid]
    result = torch.zeros(texture_height * texture_width, dtype=torch.bool, device=uv.device)
    if indices.numel():
        result[indices.unique()] = True
    return result


def distributed_visibility(
    local_visible: torch.Tensor,
    context: DistributedContext,
) -> dict[str, float]:
    """Measure local coverage, global union, and cross-rank overlap."""
    total = local_visible.numel()
    local_ratio = float(local_visible.float().mean())
    if not context.distributed:
        return {
            "visible_texel_ratio_local": local_ratio,
            "visible_texel_ratio_global": local_ratio,
            "visible_texel_overlap_ratio": 1.0,
        }
    counts = local_visible.to(torch.int32)
    local_value = torch.tensor(local_ratio, device=local_visible.device)
    dist.all_reduce(local_value, op=dist.ReduceOp.SUM)
    local_ratio = float(local_value / context.world_size)
    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    union = counts > 0
    union_count = int(union.sum())
    overlap_count = int((counts > 1).sum())
    return {
        "visible_texel_ratio_local": local_ratio,
        "visible_texel_ratio_global": union_count / total,
        "visible_texel_overlap_ratio": overlap_count / max(union_count, 1),
    }

