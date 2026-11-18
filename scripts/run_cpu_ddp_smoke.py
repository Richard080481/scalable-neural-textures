"""Launch two local Gloo workers when a Windows torchrun wheel lacks libuv."""

from __future__ import annotations

import argparse
import os
import socket
from pathlib import Path

import torch.multiprocessing as mp

from neural_texture.config import load_config
from neural_texture.train import train


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _worker(
    rank: int, world_size: int, port: int, config_path: str, output_dir: str,
    sync_each_microstep: bool,
) -> None:
    os.environ.update({
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(port),
        "RANK": str(rank),
        "LOCAL_RANK": str(rank),
        "WORLD_SIZE": str(world_size),
        "USE_LIBUV": "0",
    })
    config = load_config(config_path)
    if sync_each_microstep:
        config["training"]["ddp_sync_each_microstep"] = True
    train(config, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/smoke_test.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ddp2_cpu_smoke"))
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--sync-each-microstep", action="store_true")
    args = parser.parse_args()
    mp.spawn(
        _worker,
        args=(
            args.world_size, _free_port(), str(args.config), str(args.output_dir),
            args.sync_each_microstep,
        ),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
