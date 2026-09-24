#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:src"

# Fast software smoke test. It verifies the canonical data, the primary stream
# API, one merge sweep, feedback replay, descriptive reorganization, and
# literal-rule export. It is NOT a manuscript-scale reproduction.
python scripts/check_repo.py
python experiments/01_dataset_audit.py
python experiments/03_primary_30_streams.py --n-orders 2
python experiments/06_merge_analysis.py --n-orders 1
python experiments/10_feedback_replay.py
python experiments/12_feedback_cluster_reorganization.py
python experiments/13_literal_rules.py

echo "EvalOral smoke reproduction: PASS"
