"""Selective prediction (reject / abstain) on calibrated top-1 confidence."""

from __future__ import annotations

import numpy as np


def coverage_risk_curve(p: np.ndarray, y: np.ndarray, n_points: int = 101) -> list[dict]:
    conf = p.max(axis=1)
    correct = p.argmax(axis=1) == y
    out = []
    for t in np.linspace(0.0, 1.0, n_points):
        m = conf >= t
        cov = float(m.mean())
        out.append({
            "threshold": float(t), "coverage": cov,
            "selective_accuracy": float(correct[m].mean()) if m.any() else None,
            "risk": float(1 - correct[m].mean()) if m.any() else None,
        })
    return out


def select_threshold(p: np.ndarray, y: np.ndarray, target_selective_accuracy: float | None = None,
                     min_coverage: float | None = None) -> dict:
    """Validation-only threshold selection.

    Policy: the smallest threshold whose selective accuracy on validation reaches `target_selective_accuracy`
    (maximises coverage subject to the accuracy target). If unreachable, falls back to the threshold that
    keeps `min_coverage`. Exactly one of the two arguments must be given."""
    if (target_selective_accuracy is None) == (min_coverage is None):
        raise ValueError("give exactly one of target_selective_accuracy / min_coverage")
    conf = p.max(axis=1)
    correct = p.argmax(axis=1) == y
    order = np.argsort(-conf)
    cum_acc = np.cumsum(correct[order]) / np.arange(1, len(order) + 1)
    if target_selective_accuracy is not None:
        ok = np.where(cum_acc >= target_selective_accuracy)[0]
        if len(ok) == 0:
            return {"threshold": None, "reason": "target unreachable on validation"}
        k = ok.max()  # largest accepted set meeting the target
    else:
        k = max(0, int(np.ceil(min_coverage * len(order))) - 1)
    thr = float(conf[order][k])
    m = conf >= thr
    return {"threshold": thr, "coverage": float(m.mean()), "selective_accuracy": float(correct[m].mean()),
            "rejection_rate": float(1 - m.mean())}


def decide(p_row: np.ndarray, threshold: float | None) -> dict:
    top = int(np.argmax(p_row))
    conf = float(p_row[top])
    rejected = threshold is not None and conf < threshold
    return {"class_id": top, "confidence": conf, "rejected": bool(rejected)}


def selective_summary(p: np.ndarray, y: np.ndarray, threshold: float | None) -> dict:
    conf = p.max(axis=1)
    correct = p.argmax(axis=1) == y
    m = conf >= (threshold if threshold is not None else -1)
    return {"threshold": threshold, "accuracy_before_rejection": float(correct.mean()), "coverage": float(m.mean()),
            "selective_accuracy": float(correct[m].mean()) if m.any() else None, "rejection_rate": float(1 - m.mean())}
