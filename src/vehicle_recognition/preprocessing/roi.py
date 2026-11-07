"""ROI cropping with safe coordinate clipping and a documented padding policy.

Boxes are 0-based, half-open float pixel coordinates [x_min, y_min, x_max, y_max].
Padding policy: expand each side by `pad_frac` x box size (default 0.0 for ground-truth crops at
train/eval time; the same value MUST be used at inference). After padding, coordinates are clipped to
the image and rounded outward to integers. Degenerate boxes raise ValueError.
"""

from __future__ import annotations

import math

import numpy as np


def clip_box(box, width: int, height: int) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = (float(v) for v in box)
    if not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
        raise ValueError("Box contains non-finite values")
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    return max(0.0, x0), max(0.0, y0), min(float(width), x1), min(float(height), y1)


def pad_box(box, pad_frac: float) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = box
    pw, ph = (x1 - x0) * pad_frac, (y1 - y0) * pad_frac
    return x0 - pw, y0 - ph, x1 + pw, y1 + ph


def crop_roi(img: np.ndarray, box, pad_frac: float = 0.0, min_size: int = 2) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = clip_box(pad_box(box, pad_frac), w, h)
    ix0, iy0 = int(math.floor(x0)), int(math.floor(y0))
    ix1, iy1 = int(math.ceil(x1)), int(math.ceil(y1))
    if ix1 - ix0 < min_size or iy1 - iy0 < min_size:
        raise ValueError(f"Degenerate ROI after clipping: {(ix0, iy0, ix1, iy1)} in {w}x{h}")
    return img[iy0:iy1, ix0:ix1]


def iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0
