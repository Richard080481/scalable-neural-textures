"""Small, explicit helpers for single-node and multi-node DDP execution."""

import os
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    backend: str | None

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialize_distributed(requested_device: str = "auto") -> DistributedContext:
    """Initialize from torchrun environment variables, or return a local context."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if requested_device == "auto":
        device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    else:
        requested = torch.device(requested_device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available in this PyTorch installation.")
        device = torch.device("cuda", local_rank) if requested.type == "cuda" else requested
    backend = None
    if world_size > 1:
        backend = "nccl" if device.type == "cuda" else "gloo"
        if device.type == "cuda":
            torch.cuda.set_device(device)
        dist.init_process_group(backend=backend, init_method="env://")
    return DistributedContext(rank, local_rank, world_size, device, backend)


def cleanup_distributed(context: DistributedContext) -> None:
    if context.distributed and dist.is_initialized():
        dist.destroy_process_group()


def synchronize(context: DistributedContext) -> None:
    if context.device.type == "cuda":
        torch.cuda.synchronize(context.device)


def reduce_mean(value: torch.Tensor, context: DistributedContext) -> torch.Tensor:
    result = value.detach().clone()
    if context.distributed:
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
        result.div_(context.world_size)
    return result


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


class TimingAllReduceState:
    """Accumulate wall time spent waiting for DDP gradient bucket all-reduces."""

    def __init__(self, process_group: Any, world_size: int):
        self.process_group = process_group
        self.world_size = world_size
        self.elapsed_seconds = 0.0
        self.bytes = 0
        self._window_started: float | None = None

    def reset(self) -> None:
        self.elapsed_seconds = 0.0
        self.bytes = 0
        self._window_started = None

    def finish_window(self) -> None:
        """Finish timing after the caller has synchronized the device."""
        if self._window_started is not None:
            self.elapsed_seconds += time.perf_counter() - self._window_started
            self._window_started = None


def timing_allreduce_hook(
    state: TimingAllReduceState, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    """Equivalent to DDP's default averaged all-reduce, with timing and volume."""
    tensor = bucket.buffer()
    if state._window_started is None:
        state._window_started = time.perf_counter()
    state.bytes += tensor.numel() * tensor.element_size()
    future = dist.all_reduce(tensor, group=state.process_group, async_op=True).get_future()

    def complete(result: torch.futures.Future) -> torch.Tensor:
        reduced = result.value()[0]
        reduced.div_(state.world_size)
        return reduced

    return future.then(complete)

