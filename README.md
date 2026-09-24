# EvalOral

**An Explainable Evolving Neuro-Fuzzy Framework for Oral Image Pattern Discovery and Rule-Aware Visual Auditing**

EvalOral is a reproducibility repository for an oral-image research framework that connects:

- content-deduplicated, label-free visual structure discovery;
- a single-pass evolving neuro-fuzzy classifier with persistent rule identities;
- Gaussian antecedents, relevance-weighted coverage, and observation-dependent AND/OR/COMP behavior;
- rule birth, adaptation, support, staleness, and exemplar histories;
- a separate CNN/Grad-CAM visual-audit branch;
- human semantic feedback and leave-one-volunteer-out transfer tests;
- literal before/after fuzzy-rule inspection.

> **Research use only.** EvalOral is not a clinical diagnostic device.

> **Data licensing:** the MIT license covers project software, not the source oral photographs. See [`DATA_LICENSE_NOTICE.md`](DATA_LICENSE_NOTICE.md) before publishing or redistributing `data/raw/`.

## Scientific scope

The controlled supervised anchor task contains 121 permanent-dentition DAI photographs with explicitly released view labels (56 frontal, 65 occlusal). The primary evolving experiment uses 30 randomized volunteer-order streams. Each stream fits preprocessing only on three initialization volunteers, predicts each of the remaining seven volunteer blocks before labels are revealed, and then updates the evolving model.

The primary fuzzy model uses a lightweight handcrafted image representation rather than CNN embeddings:

```text
oral photograph
    -> grayscale 56x56 + gradient-edge 56x56
    -> concatenate (6,272 descriptors)
    -> StandardScaler
    -> PCA whitening (4 PCs)
    -> evolving Gaussian fuzzy rules
    -> relevance-weighted coverage + observation-dependent Unim
    -> class evidence (frontal / occlusal)
```

The CNN is intentionally separate and is used only for Grad-CAM auditing.

## Repository layout

```text
EvalOral_Reproducibility_Repository/
├── README.md
├── README_PTBR.md
├── pyproject.toml
├── requirements.txt
├── environment.yml
├── LICENSE
├── CITATION.cff
├── Makefile
│
├── src/evaloral/
│   ├── data.py
│   ├── features.py
│   ├── evolving_nf.py
│   ├── legacy_nf.py
│   ├── protocols.py
│   ├── clustering.py
│   ├── rule_history.py
│   ├── gradcam.py
│   ├── feedback.py
│   └── stats.py
│
├── experiments/
│   ├── 00_prepare_data.py
│   ├── 01_dataset_audit.py
│   ├── 02_unsupervised_clustering.py
│   ├── 03_primary_30_streams.py
│   ├── 04_ablation.py
│   ├── 05_hyperparameter_sensitivity.py
│   ├── 06_merge_analysis.py
│   ├── 07_rule_history.py
│   ├── 08_cluster_rule_concordance.py
│   ├── 09_gradcam.py
│   ├── 10_feedback_replay.py
│   ├── 11_feedback_generalization.py
│   ├── 12_feedback_cluster_reorganization.py
│   └── 13_literal_rules.py
│
├── data/
│   ├── raw/                 # canonical source-module image snapshot
│   ├── processed/           # manifests + 6,272-D feature matrices
│   └── reference/           # paper-linked interpretability snapshots
│
├── validation/
│   ├── paper_audit_protocol/
│   └── anatomical_hitl/     # later research extension; not a paper result
│
├── results/
│   ├── reference/           # manuscript-reported headline values
│   └── generated/           # created by experiment scripts
│
├── docs/
│   ├── PAPER_TO_CODE_MAP.md
│   ├── REPRODUCIBILITY_NOTES.md
│   ├── MODEL_SCOPE.md
│   ├── DATA_PROVENANCE.md
│   ├── EXTERNAL_VALIDATION.md
│   └── GITHUB_PUBLISH_CHECKLIST.md
│
├── paper/
│   └── EvalOral_manuscript.pdf
├── scripts/
├── tests/
└── .github/workflows/ci.yml
```

## Installation

### Python virtual environment

```bash
python -m venv .venv
```

Activate it:

```bash
# Linux / macOS
source .venv/bin/activate
```

```bat
REM Windows
.venv\Scripts\activate
```

Install the full research environment:

```bash
python -m pip install --upgrade pip
pip install -e ".[all]"
```

For everything except the CNN/Grad-CAM experiment:

```bash
pip install -e ".[dev]"
```

## Quick reproducibility checks

```bash
python scripts/check_repo.py
pytest -q
```

Audit the bundled data:

```bash
python experiments/01_dataset_audit.py
```

Expected canonical counts:

```text
source-module records:       441
content-unique photographs:  257
DAI view anchor:              121
frontal:                       56
occlusal:                      65
volunteer-order streams:       30
```

Fast end-to-end smoke reproduction:

```bash
bash scripts/reproduce_smoke.sh
```

This uses reduced repetition counts for the computationally heavier sweeps and is intended to verify installation, data integrity, and execution flow. It does not replace the manuscript-scale defaults.

## Reproduce the main experiments

### 1. Label-free visual structure

```bash
python experiments/02_unsupervised_clustering.py
```

This fits StandardScaler + non-whitened PCA(4) on the 257 content-unique photographs and evaluates K-means for `k=2...12`. The representative `k=2` and `k=7` partitions use `random_state=42` and `n_init=20`.

### 2. Primary evolving neuro-fuzzy experiment

```bash
python experiments/03_primary_30_streams.py
```

The bundled reference implementation reproduces the evolving-model headline closely/exactly in the release environment: median macro-F1 ≈ 0.93435, median accuracy ≈ 0.93494, median final rules = 9.

**Important reproducibility note:** the current paired LinearSVC rerun does not reproduce the manuscript's reported 0.964 median macro-F1. See [`docs/REPRODUCIBILITY_NOTES.md`](docs/REPRODUCIBILITY_NOTES.md) before public release.

### 3. Ablation

```bash
python experiments/04_ablation.py
```

### 4. Sigma/alpha sensitivity

```bash
python experiments/05_hyperparameter_sensitivity.py
```

This evaluates 24 operating points × 10 volunteer orders = 240 runs.

### 5. Rule merging

```bash
python experiments/06_merge_analysis.py
```

The primary paper configuration disables functional merging. This script provides a transparent sensitivity sweep around that design choice.

### 6. Rule history and galleries

```bash
python experiments/07_rule_history.py
```

### 7. Cluster–rule concordance

```bash
python experiments/08_cluster_rule_concordance.py
```

### 8. Separate CNN and Grad-CAM branch

```bash
python experiments/09_gradcam.py --force-epoch 1
```

The `--force-epoch 1` option matches the epoch selected in the reported pilot. Historical CNN weights were not retained, so regenerated saliency maps are a deterministic rerun rather than byte-identical archival heatmaps.

### 9. Human semantic feedback replay

```bash
python experiments/10_feedback_replay.py
```

The paper-linked 40-image feedback set is bundled in `data/reference/author_feedback40.csv`.

### 10. Feedback transfer to unseen volunteers

```bash
python experiments/11_feedback_generalization.py --n-orders 30
```

This includes both feedback-calibrated fuzzy-rule semantics and feedback-seeded K-means under leave-one-volunteer-out evaluation.

### 11. Descriptive before/after cluster reorganization

```bash
python experiments/12_feedback_cluster_reorganization.py
```

This intentionally uses the same 40 audited images before and after feedback to visualize membership reorganization. It is descriptive and must not be confused with the held-out experiment above.

### 12. Literal fuzzy rules

```bash
python experiments/13_literal_rules.py
```

## One-command runs

Core paper analyses that do not require PyTorch:

```bash
bash scripts/reproduce_core.sh
```

Full suite, including CNN/Grad-CAM:

```bash
bash scripts/reproduce_all.sh
```

## Human validation

The paper's prepared blinded dental-audit protocol is in:

```text
validation/paper_audit_protocol/
```

A separate anatomical human-in-the-loop extension is preserved in:

```text
validation/anatomical_hitl/
```

That extension is future-work infrastructure and should not be used to inflate the present paper's claims.

## Paper-linked reference artifacts

`data/reference/` contains:

- final maximum-coverage rule assignments for the 121-image view subset;
- the documented 40-image author-side feedback audit;
- rule-level before/after semantic summaries;
- literal Gaussian rule parameters used in the paper table;
- selected rule-history values used in the paper.

These files are intentionally labeled `reference`: they support auditing and figure reconstruction, while the prequential predictions are regenerated by the experiment scripts.

## Data and software license

The project code is released under the MIT License. The bundled image snapshot derives from upstream SBBrasil calibration/training resources. See [`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md) for the separate data-provenance/licensing note.

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). Replace the manuscript placeholder DOI with the final DOI after publication.

## Scientific integrity / release status

The repository was assembled to expose both successful reproductions and unresolved provenance issues. Before public release, review the single flagged baseline item in [`docs/GITHUB_PUBLISH_CHECKLIST.md`](docs/GITHUB_PUBLISH_CHECKLIST.md). No result is silently overwritten merely to make a rerun match the manuscript.

## Release audit

Software integrity, smoke-test status, and known historical-reconstruction caveats are recorded in [`docs/RELEASE_AUDIT.md`](docs/RELEASE_AUDIT.md).
