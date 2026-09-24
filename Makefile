PYTHON ?= python
export PYTHONPATH := src

.PHONY: install test audit primary cluster feedback smoke core all
install:
	$(PYTHON) -m pip install -e .[all]

test:
	$(PYTHON) -m pytest -q

audit:
	$(PYTHON) experiments/01_dataset_audit.py

primary:
	$(PYTHON) experiments/03_primary_30_streams.py

cluster:
	$(PYTHON) experiments/02_unsupervised_clustering.py

feedback:
	$(PYTHON) experiments/10_feedback_replay.py
	$(PYTHON) experiments/12_feedback_cluster_reorganization.py

smoke:
	bash scripts/reproduce_smoke.sh

core:
	bash scripts/reproduce_core.sh

all:
	bash scripts/reproduce_all.sh
