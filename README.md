# 🦷 EvalOral: Explainable Evolving Neuro-Fuzzy Learning for Oral Images

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![CI](https://github.com/pdecampossouza/EvalOral/actions/workflows/ci.yml/badge.svg)](https://github.com/pdecampossouza/EvalOral/actions/workflows/ci.yml)
[![Reproducible Research](https://img.shields.io/badge/Reproducibility-paper--linked-brightgreen.svg)](docs/PAPER_TO_CODE_MAP.md)
[![Human-in-the-Loop](https://img.shields.io/badge/Human--in--the--Loop-validation-orange.svg)](validation/)
[![NOVA IMS](https://img.shields.io/badge/NOVA-IMS-8A2BE2.svg)](https://novaims.unl.pt/)

---

## 🧭 Overview

**EvalOral** is the reproducibility repository for the research framework:

> **EvalOral: An Explainable Evolving Neuro-Fuzzy Framework for Oral Image Pattern Discovery and Rule-Aware Visual Auditing**  
> **Paulo Vitor Campos de Souza, Maria Isabel Villalobos, Alisson Marques da Silva**  
> 2026

The project investigates how heterogeneous oral photographs can be transformed from a collection of images into **auditable, evolving visual knowledge**.

Instead of returning only a class label, EvalOral links each prediction to a persistent fuzzy rule whose **birth, support, updates, antecedents, logical behavior, representative images, and semantic revisions can be inspected over time**.

The framework combines:

- **label-free visual structure discovery** with K-means;
- a **single-pass evolving neuro-fuzzy classifier** with persistent rule identities;
- **Gaussian fuzzy antecedents** and relevance-weighted coverage;
- observation-dependent **AND / OR / COMP** fuzzy behavior;
- **rule lineage**: birth, adaptation, support, staleness, and exemplars;
- a **separate CNN + Grad-CAM branch** for visual explanation;
- **human semantic feedback** and replay;
- **leave-one-volunteer-out transfer** to previously unseen subjects;
- literal **IF–THEN rule inspection before and after feedback**.

> ⚠️ **Research use only.** EvalOral is an experimental research framework and is not a clinical diagnostic device.

---

## 💡 Why EvalOral?

Most image classifiers answer a question such as:

> *“This photograph is occlusal with probability 0.94.”*

EvalOral is designed to support a richer question:

> *“Which local visual rule supported the decision, when did that rule emerge, which images shaped it, how stable is it, where did the visual model focus, and what changed after human feedback?”*

This makes the learned knowledge itself an object of study.

A central observation of the paper is that human feedback can reveal two very different situations:

1. **a coherent rule with the wrong semantic name**, and
2. **a genuinely mixed visual region** that still needs refinement.

That distinction is difficult to recover from accuracy alone.

---

## 🧠 Architecture at a Glance

The **primary evolving neuro-fuzzy classifier does not use CNN embeddings**. Its predictive path is intentionally lightweight and auditable:

```text
Dental photograph
      │
      ├── grayscale image 56×56
      └── gradient-edge map 56×56
                │
                ▼
        concatenate = 6,272 descriptors
                │
                ▼
           StandardScaler
                │
                ▼
        PCA whitening → 4 PCs
                │
                ▼
      evolving Gaussian fuzzy rules
                │
      ┌─────────┴──────────┐
      │                    │
 fuzzy coverage     observation-dependent
      │               AND / OR / COMP
      └─────────┬──────────┘
                ▼
       effective rule activation
                │
                ▼
       frontal / occlusal prediction
                │
                ▼
      persistent rule history + audit
```

Two complementary branches remain separate from the primary classifier:

```text
PCA representation ──► label-free K-means visual discovery

Dental photograph ──► lightweight CNN ──► Grad-CAM visual audit

Rule galleries + Grad-CAM ──► human feedback ──► semantic rule calibration
```

This separation is important: the repository should **not** be interpreted as an end-to-end CNN-to-fuzzy architecture for the reported primary experiment.

---

## 🔬 What the Study Evaluates

### Primary view-learning task

The controlled anchor experiment uses **121 permanent-dentition DAI photographs** with released view labels:

| Item | Value |
|---|---:|
| Frontal images | 56 |
| Occlusal images | 65 |
| Volunteers | 10 |
| Random volunteer-order streams | 30 |
| Median evolving-model macro-F1 | **0.934** |
| Median final fuzzy rules | **9** |

Each stream initializes preprocessing on three volunteers and then processes the remaining seven volunteer blocks **prequentially**: predict first, reveal labels next, update afterward.

### Human feedback experiments

The focused feedback study contains **40 reviewed rule-gallery exemplars** and introduces a provisional lateral semantic label.

| Analysis | Before | After feedback |
|---|---:|---:|
| Mean audited-gallery purity | 0.575 | **0.750** |
| Descriptive cluster accuracy on the same 40 images | 0.625 | **0.825** |
| Held-out volunteer K-means semantic accuracy | 0.500 | **0.575** |

The descriptive before/after clustering experiment is intentionally separate from the held-out generalization experiment.

---

## ✨ Main Research Contributions

✅ **Content-aware oral-image analysis** with SHA-256 deduplication and provenance tracking  
✅ **Label-free visual discovery** before supervised/evolving interpretation  
✅ **Single-pass evolving fuzzy learning** over volunteer streams  
✅ **Persistent rule identities** rather than disposable cluster IDs  
✅ **Literal fuzzy rules** with Gaussian centers, widths, relevance, consequent, and support  
✅ **Rule biographies**: birth, adaptation, support, logical regime, last update, and staleness  
✅ **Rule-linked Grad-CAM** for cross-layer visual auditing  
✅ **Human feedback replay** that distinguishes semantic misnaming from geometric mixing  
✅ **Leave-one-volunteer-out transfer** after feedback  
✅ **Before/after cluster reorganization** showing which images actually changed semantic group  
✅ **Blinded external dental-validation workflow**  
✅ Tests, CI, integrity checks, hashes, and paper-to-code traceability

---

## 📁 Repository Structure

```text
EvalOral/
│
├── README.md
├── README_PTBR.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── environment.yml
├── pyproject.toml
├── Makefile
│
├── src/evaloral/                  # Reusable research code
│   ├── data.py
│   ├── features.py
│   ├── evolving_nf.py
│   ├── protocols.py
│   ├── clustering.py
│   ├── rule_history.py
│   ├── gradcam.py
│   ├── feedback.py
│   └── stats.py
│
├── experiments/                   # Paper-linked experiments
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
│   ├── raw/                        # Canonical oral-image snapshot
│   ├── processed/                  # Manifests + feature matrices
│   └── reference/                  # Paper-linked audit artifacts
│
├── validation/
│   ├── paper_audit_protocol/       # Blinded dental-review workflow
│   └── anatomical_hitl/            # Future anatomical HITL extension
│
├── results/
│   ├── reference/
│   └── generated/
│
├── docs/                            # Reproducibility and provenance docs
├── paper/                           # Manuscript snapshot
├── scripts/                         # Reproduction helpers
├── tests/                           # Automated tests
└── .github/workflows/ci.yml         # Continuous integration
```

---

## ⚙️ Installation

### Option 1 — Python virtual environment

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

For the core experiments without the CNN / Grad-CAM branch:

```bash
pip install -e ".[dev]"
```

### Option 2 — Conda

```bash
conda env create -f environment.yml
conda activate evaloral
```

---

## ✅ Verify the Repository

Run the integrity checks:

```bash
python scripts/check_repo.py
```

Expected result:

```text
12 / 12 checks passed
```

Run the automated tests:

```bash
pytest -q
```

Expected release result:

```text
6 passed
```

Fast end-to-end smoke reproduction:

```bash
bash scripts/reproduce_smoke.sh
```

---

## 🧪 Reproduce the Paper Experiments

| Experiment | Command | Purpose |
|---|---|---|
| Dataset audit | `python experiments/01_dataset_audit.py` | Integrity, counts, provenance |
| Unsupervised clustering | `python experiments/02_unsupervised_clustering.py` | Label-free visual structure |
| 30 evolving streams | `python experiments/03_primary_30_streams.py` | Primary neuro-fuzzy evaluation |
| Ablation | `python experiments/04_ablation.py` | Component contribution |
| Sensitivity | `python experiments/05_hyperparameter_sensitivity.py` | `sigma × alpha` analysis |
| Rule merging | `python experiments/06_merge_analysis.py` | Merge sensitivity |
| Rule history | `python experiments/07_rule_history.py` | Rule biographies and galleries |
| Cluster–rule concordance | `python experiments/08_cluster_rule_concordance.py` | Compare unsupervised and fuzzy structure |
| Grad-CAM | `python experiments/09_gradcam.py --force-epoch 1` | Visual explanation branch |
| Feedback replay | `python experiments/10_feedback_replay.py` | Human semantic calibration |
| Feedback generalization | `python experiments/11_feedback_generalization.py --n-orders 30` | Held-out volunteer transfer |
| Cluster reorganization | `python experiments/12_feedback_cluster_reorganization.py` | Before/after membership change |
| Literal rules | `python experiments/13_literal_rules.py` | Explicit IF–THEN rule extraction |

### One-command reproduction

Core analyses:

```bash
bash scripts/reproduce_core.sh
```

Full suite, including CNN / Grad-CAM:

```bash
bash scripts/reproduce_all.sh
```

---

## 🧑‍⚕️ Human Validation

The repository includes a blinded dental-audit protocol under:

```text
validation/paper_audit_protocol/
```

The reviewer-facing workflow evaluates:

- visual cluster coherence;
- rule-linked Grad-CAM plausibility;
- artifact influence;
- fuzzy-rule history readability and usefulness.

A separate anatomical human-in-the-loop extension is stored under:

```text
validation/anatomical_hitl/
```

This extension is **future-work infrastructure** and is intentionally separated from the claims of the present paper.

---

## 🧩 From Fuzzy Rules to Dental Knowledge

A key goal of EvalOral is to make the local knowledge explicit.

A learned region can be represented conceptually as:

```text
IF PC1 ≈ c1 (σ1)
AND PC2 ≈ c2 (σ2)
AND PC3 ≈ c3 (σ3)
AND PC4 ≈ c4 (σ4)
THEN view = frontal / occlusal / feedback-calibrated lateral
```

The rule is not just a static statement. The repository also tracks:

```text
rule ID
birth step
birth reason
support
antecedent updates
AND / OR / COMP history
last update
staleness
representative images
semantic feedback
```

This enables the study of **how visual knowledge changes**, not only whether a final prediction is correct.

---

## 🔭 Research Paths Opened by EvalOral

The current paper focuses on visual structure, view learning, evolving rules, and explanation auditing. The framework is designed to support future work on:

- maxilla / mandible reasoning;
- left / right orientation;
- anterior / posterior regions;
- tooth / gingiva / other anatomical parsing;
- tooth-type discovery (incisor, canine, premolar, molar);
- tooth-level segmentation and numbering;
- PUFA, trauma, caries, and other condition-specific expert labels;
- active learning from dentist feedback;
- longitudinal person-specific evolving oral-health models;
- dedicated forensic-odontology studies.

These are research directions, **not claims demonstrated by the present paper**.

---

## 🔗 Related Dataset Pipeline

EvalOral builds on the previously released oral-image extraction and curation pipeline:

**From PDF to Dental View Classification: A Human-in-the-Loop Dataset and Pipeline for Oral Health Imaging**  
👉 https://github.com/pdecampossouza/Pipeline-for-Oral-Health-Images

That repository focuses on extracting and structuring the public oral-health images. **EvalOral focuses on what can be learned, explained, audited, and evolved from them.**

---

## 📘 Paper and Code Traceability

A direct map from manuscript components to executable scripts is available in:

[`docs/PAPER_TO_CODE_MAP.md`](docs/PAPER_TO_CODE_MAP.md)

Additional reproducibility documentation:

- [`docs/REPRODUCIBILITY_NOTES.md`](docs/REPRODUCIBILITY_NOTES.md)
- [`docs/RELEASE_AUDIT.md`](docs/RELEASE_AUDIT.md)
- [`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md)
- [`docs/MODEL_SCOPE.md`](docs/MODEL_SCOPE.md)

### Reproducibility note

The evolving-model headline is reproducible in the packaged environment. A historical LinearSVC headline value in the manuscript does not currently reproduce under the paper-matched rerun; this discrepancy is documented transparently in [`docs/REPRODUCIBILITY_NOTES.md`](docs/REPRODUCIBILITY_NOTES.md) rather than being silently overwritten.

---

## 📘 Citation

If you use EvalOral, please cite the manuscript and software repository.

```bibtex
@software{souza2026evaloral,
  author  = {Paulo Vitor Campos de Souza and Maria Isabel Villalobos and Alisson Marques da Silva},
  title   = {EvalOral: An Explainable Evolving Neuro-Fuzzy Framework for Oral Image Pattern Discovery and Rule-Aware Visual Auditing},
  year    = {2026},
  url     = {https://github.com/pdecampossouza/EvalOral},
  license = {MIT}
}
```

Citation metadata is also available in [`CITATION.cff`](CITATION.cff).

---

## ⚖️ License

The project software is released under the **MIT License**. See [`LICENSE`](LICENSE).

Data provenance and image-redistribution information are documented separately in:

- [`DATA_LICENSE_NOTICE.md`](DATA_LICENSE_NOTICE.md)
- [`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md)

---

## 📬 Contact

**Paulo Vitor Campos de Souza**  
NOVA Information Management School (NOVA IMS)  
Universidade Nova de Lisboa, Portugal  
GitHub: [@pdecampossouza](https://github.com/pdecampossouza)

---

> ✳️ **From oral photographs to evolving, auditable dental-image knowledge.**
