from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image


def compute_feature(path: str | Path, size: int = 224, down_factor: int = 4) -> np.ndarray:
    """Exact lightweight representation used by SBBrasil TrainSheets/EvalOral.

    RGB -> grayscale + NumPy gradient magnitude -> block mean downsample to
    56x56 for size=224/down_factor=4 -> concatenate -> 6,272 values.
    """
    path = Path(path)
    with Image.open(path) as im:
        im = im.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    gx = np.gradient(gray, axis=1)
    gy = np.gradient(gray, axis=0)
    edge = np.hypot(gx, gy)

    def down(a: np.ndarray) -> np.ndarray:
        h, w = a.shape
        f = down_factor
        hh, ww = h // f, w // f
        return a[: hh * f, : ww * f].reshape(hh, f, ww, f).mean(axis=(1, 3))

    g = down(gray).astype(np.float32)
    e = down(edge).astype(np.float32)
    return np.concatenate([g.ravel(), e.ravel()]).astype(np.float32)
