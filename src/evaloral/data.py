from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_source(root: Path | None = None):
    root = Path(root) if root else repo_root()
    meta = pd.read_csv(root / "data/processed/source_manifest.csv")
    X = np.load(root / "data/processed/features_source441.npy")
    return X, meta


def load_unique(root: Path | None = None):
    root = Path(root) if root else repo_root()
    meta = pd.read_csv(root / "data/processed/content_unique_manifest.csv")
    X = np.load(root / "data/processed/features_unique257.npy")
    return X, meta


def load_view121(root: Path | None = None):
    root = Path(root) if root else repo_root()
    meta = pd.read_csv(root / "data/processed/view121_manifest.csv")
    X = np.load(root / "data/processed/features_view121.npy")
    y = meta["released_view"].map({"frontal": 0, "occlusal": 1}).to_numpy(dtype=int)
    return X, y, meta


def image_path(relative: str, root: Path | None = None) -> Path:
    root = Path(root) if root else repo_root()
    return root / relative


def dataset_summary(root: Path | None = None) -> dict:
    root = Path(root) if root else repo_root()
    return json.loads((root / "data/processed/dataset_summary.json").read_text(encoding="utf-8"))
