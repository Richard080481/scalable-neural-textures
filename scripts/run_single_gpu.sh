#!/bin/bash
set -euo pipefail
python -m neural_texture.train --config "${1:-configs/baseline.yaml}" --output-dir "${2:-outputs/single_gpu}"

