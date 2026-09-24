# EvalOral v1.0.0 — reproducibility release candidate

This release candidate consolidates the code, canonical image snapshot, processed features, paper-linked reference artifacts, validation protocols, and future anatomical HITL extension used across the EvalOral research workflow.

## Included

- 441 canonical source-module image records;
- SHA-256 content deduplication and 257-image label-free corpus;
- 121-image DAI frontal/occlusal anchor task;
- 6,272-D grayscale + gradient-edge feature extraction;
- 30 volunteer-order evolving neuro-fuzzy protocol;
- static LinearSVC anchor rerun;
- methodological ablation;
- sigma/alpha sensitivity grid;
- rule-merging sensitivity;
- persistent rule-history monitoring and galleries;
- K-means `k=2...12` visual-structure analysis;
- cluster–rule concordance;
- separate CNN + Grad-CAM audit branch;
- 40-image author-side semantic feedback replay;
- leave-one-volunteer-out feedback transfer;
- feedback-seeded clustering;
- before/after cluster-membership reorganization;
- literal fuzzy rules before/after feedback;
- blinded dental-audit protocol;
- reviewer/admin anatomical HITL extension packages;
- tests, CI, integrity checks, and release hashes.

## Scientific caveats retained intentionally

The release does not hide historical reconstruction gaps. In particular, the current paired LinearSVC rerun gives approximately 0.939 median macro-F1 rather than the manuscript's 0.964 value, and the historical intermediate ablation scripts were not retained as immutable artifacts. See `docs/REPRODUCIBILITY_NOTES.md` and `docs/RELEASE_AUDIT.md`.

## Public-release gate

Before publishing the raw-image snapshot, confirm source/public-data redistribution conditions. See `DATA_LICENSE_NOTICE.md`.
