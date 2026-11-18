#!/bin/bash
set -euo pipefail
GPUS="${1:-2}"
CONFIG="${2:-configs/scaling.yaml}"
OUTPUT="${3:-outputs/ddp_${GPUS}gpu}"
torchrun --standalone --nproc_per_node="$GPUS" -m neural_texture.train --config "$CONFIG" --output-dir "$OUTPUT"

