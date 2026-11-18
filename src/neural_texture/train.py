"""Single-process or torchrun DDP neural-texture training entry point."""

from __future__ import annotations

import argparse
import contextlib
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

from .config import load_config, save_config
from .data import create_dataset
from .distributed import (
    DistributedContext,
    TimingAllReduceState,
    cleanup_distributed,
    initialize_distributed,
    reduce_mean_values,
    synchronize,
    timing_allreduce_hook,
)
from .metrics import distributed_visibility, masked_l1, masked_psnr, visible_texel_mask
from .models import NeuralTextureModel
from .utils.checkpoint import load_checkpoint, save_checkpoint
from .utils.logging import CSVLogger, environment_metadata, write_json
from .utils.reproducibility import seed_everything
from .visualization import create_progress_gif, save_progress_frame


METRIC_FIELDS = [
    "step", "epoch", "world_size", "global_batch_size", "micro_batch_size",
    "gradient_accumulation_steps", "loss", "psnr", "step_time_ms", "data_time_ms",
    "forward_time_ms", "backward_time_ms", "communication_time_ms", "optimizer_time_ms",
    "samples_per_second", "communication_bytes", "visible_texel_ratio_local",
    "visible_texel_ratio_global", "visible_texel_overlap_ratio", "allocated_memory_mb",
    "reserved_memory_mb", "device",
]


def choose_device(requested: str) -> torch.device:
    """Backward-compatible local device selection used by evaluation."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this PyTorch installation.")
    return device


def _max_across_ranks(values: list[float], context: DistributedContext) -> list[float]:
    tensor = torch.tensor(values, dtype=torch.float64, device=context.device)
    if context.distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor.cpu().tolist()


def _batch_plan(config: dict[str, Any], context: DistributedContext) -> tuple[int, int, int]:
    training = config["training"]
    global_batch = training.get("global_batch_size") or training["batch_size"]
    accumulation = training["gradient_accumulation_steps"]
    divisor = context.world_size * accumulation
    if global_batch % divisor:
        raise ValueError(
            f"global_batch_size={global_batch} must be divisible by world_size={context.world_size} "
            f"times gradient_accumulation_steps={accumulation}."
        )
    return global_batch, global_batch // divisor, accumulation


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) not in (None, "")]
    return statistics.fmean(values) if values else None


@torch.no_grad()
def _validate(
    model: NeuralTextureModel,
    config: dict[str, Any],
    context: DistributedContext,
    output: Path,
) -> float:
    """Evaluate disjoint validation views and reduce a global masked PSNR."""
    dataset = create_dataset(config, "val")
    indices = list(range(context.rank, len(dataset), context.world_size))
    loader = DataLoader(Subset(dataset, indices), batch_size=1, shuffle=False, num_workers=0)
    squared_error = torch.zeros((), dtype=torch.float64, device=context.device)
    valid_values = torch.zeros((), dtype=torch.float64, device=context.device)
    first_batch: dict[str, Any] | None = None
    model.eval()
    for batch in loader:
        rgb = batch["rgb"].to(context.device)
        uv = batch["uv"].to(context.device)
        mask = batch["mask"].to(context.device)
        prediction = model(uv, mask)
        squared_error += (((prediction - rgb) ** 2) * mask).double().sum()
        valid_values += mask.double().sum() * prediction.shape[1]
        if first_batch is None:
            first_batch = {"rgb": rgb, "uv": uv, "mask": mask}
    totals = torch.stack((squared_error, valid_values))
    if context.distributed:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    mse = totals[0] / totals[1].clamp_min(1.0)
    psnr = float(-10.0 * torch.log10(mse.clamp_min(1e-12)))
    if context.is_main and first_batch is not None:
        save_progress_frame(
            model, first_batch["uv"], first_batch["mask"], first_batch["rgb"],
            config["training"]["steps"], output / "renders" / "validation",
        )
    return psnr


def train(config: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    context = initialize_distributed(config["device"])
    seed_everything(config["seed"] + context.rank)
    device = context.device
    output = Path(output_dir)
    logger: CSVLogger | None = None
    profiler: torch.profiler.profile | None = None
    completed = False
    try:
        if context.is_main:
            for directory in (output, output / "checkpoints", output / "renders", output / "logs", output / "profiler"):
                directory.mkdir(parents=True, exist_ok=True)
            save_config(config, output / "config.yaml")
            metadata = environment_metadata()
            metadata.update({"world_size": context.world_size, "backend": context.backend})
            write_json(output / "environment.json", metadata)
        if context.distributed:
            dist.barrier()

        global_batch, micro_batch, accumulation = _batch_plan(config, context)
        dataset = create_dataset(config, "train")
        if len(dataset) < micro_batch:
            raise ValueError(f"Dataset has {len(dataset)} views but each rank needs a micro-batch of {micro_batch}.")
        # Use the same sampler at world size one so every scaling run observes
        # an identical shuffled global sample order.
        sampler = DistributedSampler(
            dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
            seed=config["seed"],
            drop_last=False,
        )
        loader = DataLoader(
            dataset,
            batch_size=micro_batch,
            sampler=sampler,
            shuffle=False,
            num_workers=config["training"]["num_workers"],
            drop_last=True,
            pin_memory=device.type == "cuda",
            generator=torch.Generator().manual_seed(config["seed"] + context.rank),
        )
        torch.backends.cudnn.benchmark = bool(
            config["training"]["cudnn_benchmark"] and device.type == "cuda"
        )
        raw_model = NeuralTextureModel.from_config(config).to(device)
        use_fused_optimizer = bool(config["training"]["fused_optimizer"] and device.type == "cuda")
        optimizer = torch.optim.Adam(
            raw_model.parameters(),
            lr=config["training"]["learning_rate"],
            fused=use_fused_optimizer,
        )
        start_step = 0
        if config["training"].get("resume"):
            state = load_checkpoint(config["training"]["resume"], raw_model, optimizer, device)
            start_step = int(state["step"])

        model: torch.nn.Module = raw_model
        communication_state: TimingAllReduceState | None = None
        if context.distributed:
            model = DDP(
                raw_model,
                device_ids=[context.local_rank] if device.type == "cuda" else None,
                output_device=context.local_rank if device.type == "cuda" else None,
            )
            if config["training"]["measure_communication"]:
                communication_state = TimingAllReduceState(dist.group.WORLD, context.world_size)
                model.register_comm_hook(communication_state, timing_allreduce_hook)

        use_amp = bool(config["training"]["mixed_precision"] and device.type == "cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        if context.is_main:
            logger = CSVLogger(output / "metrics.csv", METRIC_FIELDS)

        visual_batch = None
        if context.is_main:
            visual_batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))
            visual_batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in visual_batch.items()}
            save_progress_frame(raw_model, visual_batch["uv"], visual_batch["mask"], visual_batch["rgb"], start_step, output / "renders")

        epoch = 0
        sampler.set_epoch(epoch)
        iterator = iter(loader)
        rows: list[dict[str, Any]] = []
        initial_loss: float | None = None
        final_loss: float | None = None
        started = time.perf_counter()
        model.train()
        profiling = config["profiling"]
        if profiling["enabled"]:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            profiler = torch.profiler.profile(
                activities=activities,
                schedule=torch.profiler.schedule(
                    wait=profiling["wait_steps"],
                    warmup=profiling["warmup_steps"],
                    active=profiling["active_steps"],
                    repeat=1,
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    str(output / "profiler"), worker_name=f"rank{context.rank}"
                ),
                record_shapes=profiling["record_shapes"],
                profile_memory=True,
                with_stack=False,
            )
            profiler.start()

        for step in range(start_step + 1, config["training"]["steps"] + 1):
            step_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            if communication_state is not None:
                communication_state.reset()
            totals = {"data": 0.0, "forward": 0.0, "backward": 0.0}
            local_loss = torch.zeros((), device=device)
            local_psnr = torch.zeros((), device=device)
            visibility_every = config["training"]["visibility_every"]
            measure_visibility = step == 1 or step % visibility_every == 0
            local_visible = None
            if measure_visibility:
                local_visible = torch.zeros(
                    config["model"]["texture_height"] * config["model"]["texture_width"],
                    dtype=torch.bool,
                    device=device,
                )

            for micro_step in range(accumulation):
                data_started = time.perf_counter()
                try:
                    batch = next(iterator)
                except StopIteration:
                    epoch += 1
                    sampler.set_epoch(epoch)
                    iterator = iter(loader)
                    batch = next(iterator)
                rgb = batch["rgb"].to(device, non_blocking=True)
                uv = batch["uv"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                if config["training"]["synchronize_each_phase"]:
                    synchronize(context)
                totals["data"] += time.perf_counter() - data_started
                if local_visible is not None:
                    local_visible |= visible_texel_mask(
                        uv, mask, config["model"]["texture_height"], config["model"]["texture_width"]
                    )

                sync_gradients = (
                    config["training"]["ddp_sync_each_microstep"]
                    or micro_step == accumulation - 1
                )
                sync_context = contextlib.nullcontext()
                if context.distributed and not sync_gradients:
                    sync_context = model.no_sync()  # type: ignore[union-attr]
                with sync_context:
                    forward_started = time.perf_counter()
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                        prediction = model(uv, mask)
                        unscaled_loss = masked_l1(prediction, rgb, mask)
                        loss = unscaled_loss / accumulation
                    if config["training"]["synchronize_each_phase"]:
                        synchronize(context)
                    totals["forward"] += time.perf_counter() - forward_started
                    local_loss += unscaled_loss.detach() / accumulation
                    local_psnr += masked_psnr(prediction.detach(), rgb, mask) / accumulation
                    backward_started = time.perf_counter()
                    scaler.scale(loss).backward()
                if config["training"]["synchronize_each_phase"]:
                    synchronize(context)
                if communication_state is not None and sync_gradients:
                    communication_state.finish_window()
                totals["backward"] += time.perf_counter() - backward_started

            optimizer_started = time.perf_counter()
            scaler.step(optimizer)
            scaler.update()
            synchronize(context)
            optimizer_elapsed = time.perf_counter() - optimizer_started
            step_elapsed = time.perf_counter() - step_started
            communication_elapsed = communication_state.elapsed_seconds if communication_state else 0.0
            communication_bytes = communication_state.bytes if communication_state else 0

            reduced_metrics = reduce_mean_values(torch.stack((local_loss, local_psnr)), context)
            loss_value, psnr_value = (float(value) for value in reduced_metrics)
            initial_loss = loss_value if initial_loss is None else initial_loss
            final_loss = loss_value
            timings = _max_across_ranks(
                [step_elapsed, totals["data"], totals["forward"], totals["backward"], communication_elapsed, optimizer_elapsed],
                context,
            )
            step_elapsed, data_elapsed, forward_elapsed, backward_elapsed, communication_elapsed, optimizer_elapsed = timings
            if not config["training"]["synchronize_each_phase"]:
                # Only end-to-end step timing is meaningful in throughput mode;
                # phase launch times would otherwise be mistaken for GPU durations.
                data_elapsed = forward_elapsed = backward_elapsed = optimizer_elapsed = 0.0
            visibility: dict[str, float | str] = {key: "" for key in (
                "visible_texel_ratio_local", "visible_texel_ratio_global", "visible_texel_overlap_ratio"
            )}
            if local_visible is not None:
                visibility = distributed_visibility(local_visible, context)
            allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0
            reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2) if device.type == "cuda" else 0.0
            row = {
                "step": step, "epoch": epoch, "world_size": context.world_size,
                "global_batch_size": global_batch, "micro_batch_size": micro_batch,
                "gradient_accumulation_steps": accumulation, "loss": loss_value, "psnr": psnr_value,
                "step_time_ms": step_elapsed * 1000, "data_time_ms": data_elapsed * 1000,
                "forward_time_ms": forward_elapsed * 1000, "backward_time_ms": backward_elapsed * 1000,
                "communication_time_ms": communication_elapsed * 1000,
                "optimizer_time_ms": optimizer_elapsed * 1000,
                "samples_per_second": global_batch / step_elapsed,
                "communication_bytes": communication_bytes, **visibility,
                "allocated_memory_mb": allocated, "reserved_memory_mb": reserved, "device": str(device),
            }
            if context.is_main:
                assert logger is not None
                logger.write(row)
                rows.append(row)
                if step == 1 or step % config["training"]["log_every"] == 0:
                    print(
                        f"step={step:05d} world={context.world_size} loss={loss_value:.6f} "
                        f"psnr={psnr_value:.2f} throughput={row['samples_per_second']:.1f} samples/s "
                        f"comm_ms={row['communication_time_ms']:.2f}"
                    )
                if visual_batch is not None and (step % config["training"]["render_every"] == 0 or step == config["training"]["steps"]):
                    save_progress_frame(raw_model, visual_batch["uv"], visual_batch["mask"], visual_batch["rgb"], step, output / "renders")
                interval = config["training"]["checkpoint_every"]
                if step % interval == 0 or step == config["training"]["steps"]:
                    save_checkpoint(output / "checkpoints" / f"step_{step:06d}.pt", raw_model, optimizer, step)
            if profiler is not None:
                profiler.step()

        if profiler is not None:
            profiler.stop()
            profiler = None
        validation_psnr = _validate(raw_model, config, context, output)
        if context.distributed:
            dist.barrier()
        wall_time = time.perf_counter() - started
        summary: dict[str, Any] = {}
        if context.is_main:
            assert logger is not None
            logger.close()
            logger = None
            create_progress_gif(output / "renders", output / "renders" / "training_progress.gif")
            warmup = config["training"]["warmup_steps"]
            timed_rows = [row for row in rows if int(row["step"]) > warmup] or rows
            texture_parameters = (
                config["model"]["texture_height"] * config["model"]["texture_width"]
                * config["model"]["texture_channels"]
            )
            mean_step_time_ms = _mean(timed_rows, "step_time_ms")
            mean_samples_per_second = (
                global_batch * 1000.0 / mean_step_time_ms
                if mean_step_time_ms is not None and mean_step_time_ms > 0
                else None
            )
            summary = {
                "steps": config["training"]["steps"], "world_size": context.world_size,
                "global_batch_size": global_batch, "micro_batch_size": micro_batch,
                "gradient_accumulation_steps": accumulation,
                "ddp_sync_each_microstep": config["training"]["ddp_sync_each_microstep"],
                "synchronize_each_phase": config["training"]["synchronize_each_phase"],
                "measure_communication": config["training"]["measure_communication"],
                "fused_optimizer": use_fused_optimizer,
                "channels_last": config["model"]["channels_last"],
                "cudnn_benchmark": config["training"]["cudnn_benchmark"],
                "phase_timings_available": config["training"]["synchronize_each_phase"],
                "texture_parameters": texture_parameters,
                "texture_gradient_bytes_fp32": texture_parameters * 4,
                "initial_loss": initial_loss, "final_loss": final_loss,
                "loss_reduction": None if initial_loss is None or final_loss is None else initial_loss - final_loss,
                "wall_time_seconds": wall_time, "mean_step_time_ms": mean_step_time_ms,
                "mean_communication_time_ms": _mean(timed_rows, "communication_time_ms"),
                "mean_communication_bytes": _mean(timed_rows, "communication_bytes"),
                "mean_samples_per_second": mean_samples_per_second,
                "final_psnr": rows[-1]["psnr"] if rows else None,
                "validation_psnr": validation_psnr,
                "device": str(device),
            }
            write_json(output / "summary.json", summary)
        completed = True
        return summary
    finally:
        if logger is not None:
            logger.close()
        if profiler is not None:
            profiler.stop()
        if context.distributed:
            if completed:
                dist.barrier()
            cleanup_distributed(context)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, help="Checkpoint path; overrides training.resume in YAML")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.resume is not None:
        config["training"]["resume"] = str(args.resume)
    train(config, args.output_dir)


if __name__ == "__main__":
    main()
