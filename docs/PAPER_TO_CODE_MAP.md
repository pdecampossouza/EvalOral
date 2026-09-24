# Paper-to-code map

| Paper component | Code / data |
|---|---|
| Corpus audit and SHA-256 deduplication | `experiments/01_dataset_audit.py`, `data/processed/*manifest.csv` |
| Grayscale + edge 6,272-D descriptors | `src/evaloral/features.py`, `experiments/00_prepare_data.py` |
| Primary 30 volunteer-order streams | `experiments/03_primary_30_streams.py`, `src/evaloral/protocols.py` |
| Evolving Gaussian fuzzy rules / coverage / Unim | `src/evaloral/evolving_nf.py` |
| Static LinearSVC anchor | `src/evaloral/protocols.py::run_static_anchor` |
| Methodological ablation | `experiments/04_ablation.py` |
| Sigma × alpha sensitivity (24 × 10 runs) | `experiments/05_hyperparameter_sensitivity.py` |
| Rule-merging analysis | `experiments/06_merge_analysis.py` |
| Persistent rule history / final galleries | `experiments/07_rule_history.py`, `data/reference/rule_history_table6.csv` |
| Label-free k=2…12 clustering | `experiments/02_unsupervised_clustering.py` |
| Fine-cluster ↔ final-rule concordance | `experiments/08_cluster_rule_concordance.py` |
| Separate CNN + Grad-CAM | `experiments/09_gradcam.py`, `src/evaloral/gradcam.py` |
| 40-image human semantic replay | `experiments/10_feedback_replay.py`, `data/reference/author_feedback40.csv` |
| Leave-one-volunteer-out feedback transfer | `experiments/11_feedback_generalization.py` |
| Feedback-seeded K-means generalization | `experiments/11_feedback_generalization.py` |
| Same-40 before/after cluster reorganization | `experiments/12_feedback_cluster_reorganization.py` |
| Literal fuzzy rules before/after feedback | `experiments/13_literal_rules.py`, `data/reference/literal_rules_table10.csv` |
| Prepared blinded dental audit | `validation/paper_audit_protocol/` |
| Anatomical HITL extension | `validation/anatomical_hitl/` (not part of manuscript claims) |
