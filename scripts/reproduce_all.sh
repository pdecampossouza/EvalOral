#!/usr/bin/env bash
set -euo pipefail
bash scripts/reproduce_core.sh
export PYTHONPATH="${PYTHONPATH:-}:src"
python experiments/09_gradcam.py --force-epoch 1
