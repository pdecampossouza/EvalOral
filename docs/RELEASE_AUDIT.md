# Release audit

This file records software-integrity checks performed while assembling the GitHub release candidate.

## Repository integrity

`python scripts/check_repo.py` passed all 12 bundled-data checks:

- 441 source-module records;
- 257 content-unique photographs;
- 121 DAI view-labelled images;
- 56 frontal and 65 occlusal labels;
- expected 6,272-D feature-matrix shapes;
- 40 feedback exemplars with the expected 12/14/14 frontal/lateral/occlusal distribution;
- no missing canonical raw files;
- no SHA-256 mismatch among the 121 anchor images.

## Unit tests

`pytest -q` passed: **6 passed**.

## One-command smoke reproduction

`bash scripts/reproduce_smoke.sh` completed successfully in the release environment after the final script adjustments.

## Experiment smoke checks

The following scripts were executed successfully during release assembly:

- `01_dataset_audit.py`;
- `02_unsupervised_clustering.py`;
- `03_primary_30_streams.py` at manuscript scale (30 orders);
- `04_ablation.py` at manuscript scale (30 orders);
- `05_hyperparameter_sensitivity.py` at manuscript scale (24 operating points × 10 orders);
- `06_merge_analysis.py --n-orders 1` as a software smoke test;
- `07_rule_history.py`;
- `08_cluster_rule_concordance.py`;
- `09_gradcam.py --force-epoch 1 --max-selection-epochs 2` as a CNN/Grad-CAM software smoke test;
- `10_feedback_replay.py`;
- `11_feedback_generalization.py --n-orders 2` as a software smoke test;
- `12_feedback_cluster_reorganization.py`;
- `13_literal_rules.py`.

The full defaults of the merging and feedback-generalization sweeps are intentionally retained in the scripts. They are computationally heavier than the smoke settings above.

## Important scientific reconciliation items

This is a release candidate, not a claim that every historical manuscript number is byte-for-byte reconstructable.

1. The primary evolving neuro-fuzzy rerun reproduces the manuscript headline closely: median macro-F1 ≈ 0.93435, median accuracy ≈ 0.93494, median final rules = 9.
2. The paired LinearSVC rerun under the protocol currently described in the manuscript yields median macro-F1 ≈ 0.939 rather than the manuscript value 0.964. The upstream repository separately mentions “96.4% accuracy” for an older baseline, but the recovered upstream training script does not establish that this number is the same 30-stream macro-F1 endpoint. Do not silently equate these quantities.
3. The ablation script is a clean reconstruction of the successive components described in the paper; historical one-off intermediate scripts were not preserved, so its numerical deltas need not be identical to the manuscript table.
4. Historical CNN weights were not retained; the Grad-CAM script is a deterministic retraining protocol, not byte-identical restoration of archived heatmaps.

These points are also documented in `docs/REPRODUCIBILITY_NOTES.md` and should remain visible in a public release.
