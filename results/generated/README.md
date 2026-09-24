# Generated results

This directory is intentionally empty in the GitHub release except for this README.
Experiment scripts write fresh CSV/JSON/PNG artifacts here.

Use:

```bash
bash scripts/reproduce_smoke.sh   # fast software check
bash scripts/reproduce_core.sh    # manuscript-scale non-CNN analyses
bash scripts/reproduce_all.sh     # also retrains the separate CNN/Grad-CAM branch
```

Paper-linked archival/reference values that should not be overwritten by a rerun are stored in `../reference/` and `data/reference/`.
