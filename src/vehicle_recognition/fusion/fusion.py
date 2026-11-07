"""Hybrid probability fusion and validation-only alpha selection.

P_hybrid = alpha * P_DL + (1 - alpha) * P_CV, where both inputs are calibrated 196-class probability
matrices with identical canonical column order. alpha is selected on validation data only.
"""

from __future__ import annotations

import numpy as np

from ..calibration.calibration import nll
from ..evaluation.metrics import check_probs


class FusionUnavailableError(RuntimeError):
    """Raised when hybrid fusion is requested but a branch is unavailable - never silently degrade."""


def fuse(p_dl: np.ndarray | None, p_cv: np.ndarray | None, alpha: float) -> np.ndarray:
    if p_dl is None or p_cv is None:
        raise FusionUnavailableError("Hybrid fusion requires both deep and classical probabilities")
    if p_dl.shape != p_cv.shape:
        raise ValueError(f"shape mismatch {p_dl.shape} vs {p_cv.shape}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    check_probs(p_dl)
    check_probs(p_cv)
    out = alpha * p_dl + (1.0 - alpha) * p_cv
    return out / out.sum(axis=1, keepdims=True)


def select_alpha(p_dl: np.ndarray, p_cv: np.ndarray, y: np.ndarray, grid=None, criterion: str = "nll") -> dict:
    """Grid-search alpha on VALIDATION data. criterion: 'nll' (default; smooth, proper scoring rule) or 'top1'.
    Ties are broken towards larger alpha (simpler / closer to deep-only)."""
    grid = np.round(np.linspace(0, 1, 21), 3) if grid is None else np.asarray(grid)
    rows = []
    for a in grid:
        p = fuse(p_dl, p_cv, float(a))
        rows.append({"alpha": float(a), "top1": float(np.mean(p.argmax(1) == y)), "nll": nll(p, y)})
    if criterion == "nll":
        best = min(rows, key=lambda r: (round(r["nll"], 6), -r["alpha"]))
    else:
        best = max(rows, key=lambda r: (round(r["top1"], 6), r["alpha"]))
    return {"alpha": best["alpha"], "criterion": criterion, "grid": rows}
