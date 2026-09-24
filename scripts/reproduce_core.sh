#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:src"
python scripts/check_repo.py
python experiments/01_dataset_audit.py
python experiments/02_unsupervised_clustering.py
python experiments/03_primary_30_streams.py
python experiments/04_ablation.py
python experiments/05_hyperparameter_sensitivity.py
python experiments/06_merge_analysis.py
python experiments/07_rule_history.py
python experiments/08_cluster_rule_concordance.py
python experiments/10_feedback_replay.py
python experiments/11_feedback_generalization.py --n-orders 30
python experiments/12_feedback_cluster_reorganization.py
python experiments/13_literal_rules.py
