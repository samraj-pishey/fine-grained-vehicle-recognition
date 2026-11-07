"""Probability calibration utilities.

ECE definition used throughout: top-label (confidence) ECE with 15 equal-width bins over [0, 1];
ECE = sum_b (n_b / N) * |acc_b - conf_b|. Brier score is the multi-class version (mean squared error of the
probability vector against one-hot, summed over classes). NLL is mean -log p(y).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax

ECE_BINS = 15


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = ECE_BINS) -> float:
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def reliability_bins(p: np.ndarray, y: np.ndarray, n_bins: int = ECE_BINS) -> list[dict]:
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        out.append({
            "lo": float(edges[b]), "hi": float(edges[b + 1]), "count": int(m.sum()),
            "accuracy": float(correct[m].mean()) if m.any() else None,
            "confidence": float(conf[m].mean()) if m.any() else None,
        })
    return out


def nll(p: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], eps, 1.0))))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def calibration_report(p: np.ndarray, y: np.ndarray) -> dict:
    return {"ece_15": ece(p, y), "nll": nll(p, y), "brier": brier(p, y), "mean_confidence": float(p.max(axis=1).mean())}


def fit_temperature(logits: np.ndarray, y: np.ndarray, bounds=(0.05, 20.0)) -> float:
    """Single scalar T > 0 minimising validation NLL of softmax(logits / T) (Guo et al., 2017)."""
    logits = np.asarray(logits, dtype=np.float64)

    def loss(log_t):
        lp = log_softmax(logits / np.exp(log_t), axis=1)
        return -np.mean(lp[np.arange(len(y)), y])

    res = minimize_scalar(loss, bounds=(np.log(bounds[0]), np.log(bounds[1])), method="bounded")
    return float(np.exp(res.x))


def apply_temperature(logits: np.ndarray, t: float) -> np.ndarray:
    return softmax(np.asarray(logits, dtype=np.float64) / t, axis=1)


def probs_to_logits(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Log-probabilities act as logits (softmax(log p) == p); used to temperature-scale probability outputs."""
    return np.log(np.clip(p, eps, 1.0))
