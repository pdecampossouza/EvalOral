# Reproducibility notes

This repository distinguishes three kinds of artifacts so that a rerun is not confused with a manuscript reference snapshot.

## A. Directly rerunnable from bundled data

- Source-record/content-deduplication audit.
- 6,272-D grayscale + gradient-edge representation.
- Label-free PCA/K-means sweep and prototype extraction.
- Thirty volunteer-order evolving neuro-fuzzy streams.
- Sigma/alpha sensitivity grid.
- Rule-merging sensitivity sweep.
- Feedback semantic replay.
- Leave-one-volunteer-out feedback-calibration experiment.
- Feedback-seeded K-means and descriptive before/after membership reorganization.
- Literal-rule and rule-history visualizations from the manuscript reference tables.
- Separate CNN retraining and Grad-CAM generation.

## B. Reference snapshots recovered from the manuscript workflow

`data/reference/` contains the final maximum-coverage rule assignments, the documented 40-image author-side feedback audit, manuscript rule-history rows, and literal-rule parameters. These files make the reported interpretability analyses auditable without pretending that a post-hoc gallery is a prequential prediction.

## C. Historical components reconstructed from the current paper specification

The ablation script is a clean reimplementation of the successive methodological components described in the manuscript. Historical intermediate ad-hoc scripts were not preserved as one immutable pipeline, so small numeric deviations from the paper's ablation table are possible. The code labels this explicitly rather than presenting reconstructed outputs as archival originals.

## Static LinearSVC baseline: author verification requested before release

The current paper reports a median macro-F1 of **0.964** for the static LinearSVC anchor. The paper-aligned paired rerun in `experiments/03_primary_30_streams.py`, using the same 30 volunteer orders, three initialization volunteers, StandardScaler, four whitened PCs, and the stated LinearSVC parameters, currently produces a lower median (approximately **0.939** in the environment used to assemble this release).

The evolving neuro-fuzzy side of the same rerun reproduces the manuscript headline essentially exactly (median macro-F1 ≈ **0.93435**, median accuracy ≈ **0.93494**, median **9** final rules).

For scientific transparency, this repository stores both the rerun result and the manuscript-reported 0.964 reference instead of silently forcing the code to match the paper. The upstream repository README also mentions “96.4% accuracy” for an older view-classification baseline, but the recovered upstream script uses a different volunteer split and PCA(128), and its current rerun does not establish that this historical accuracy is the same quantity as the manuscript's paired 30-stream macro-F1. The two numbers must therefore not be silently equated. Before public release, the authors should either restore the exact historical protocol behind 0.964 or amend/clarify the manuscript endpoint.

## Grad-CAM weights

The historical CNN state used for the manuscript pilot was not retained in the recovered artifacts. `experiments/09_gradcam.py` therefore supplies the full deterministic architecture/training protocol and can force the manuscript-selected epoch (`--force-epoch 1`), but regenerated heatmaps are a reproducible rerun rather than byte-identical copies of historical pilot maps.
