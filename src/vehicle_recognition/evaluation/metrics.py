"""Classification metrics over canonical 196-class probability matrices."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score


def check_probs(p: np.ndarray, num_classes: int | None = None, atol: float = 1e-4) -> None:
    if p.ndim != 2:
        raise ValueError(f"probabilities must be 2-D, got {p.shape}")
    if num_classes is not None and p.shape[1] != num_classes:
        raise ValueError(f"expected {num_classes} columns, got {p.shape[1]}")
    if not np.all(np.isfinite(p)):
        raise ValueError("non-finite probabilities")
    if (p < -1e-9).any():
        raise ValueError("negative probabilities")
    if not np.allclose(p.sum(axis=1), 1.0, atol=atol):
        raise ValueError("rows do not sum to 1")


def topk_accuracy(p: np.ndarray, y: np.ndarray, k: int) -> float:
    topk = np.argsort(-p, axis=1)[:, :k]
    return float(np.mean((topk == y[:, None]).any(axis=1)))


def per_class_accuracy(pred: np.ndarray, y: np.ndarray, num_classes: int) -> np.ndarray:
    acc = np.full(num_classes, np.nan)
    for c in range(num_classes):
        m = y == c
        if m.any():
            acc[c] = float(np.mean(pred[m] == c))
    return acc


def classification_report(p: np.ndarray, y: np.ndarray, num_classes: int = 196) -> dict:
    check_probs(p, num_classes)
    pred = p.argmax(axis=1)
    pca = per_class_accuracy(pred, y, num_classes)
    return {
        "n": int(len(y)),
        "top1": float(np.mean(pred == y)),
        "top5": topk_accuracy(p, y, 5),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=np.arange(num_classes), zero_division=0)),
        "mean_per_class_acc": float(np.nanmean(pca)),
        "worst_class_acc": float(np.nanmin(pca)),
    }


def confusion(p_or_pred: np.ndarray, y: np.ndarray, num_classes: int = 196) -> np.ndarray:
    pred = p_or_pred.argmax(axis=1) if p_or_pred.ndim == 2 else p_or_pred
    return confusion_matrix(y, pred, labels=np.arange(num_classes))


def top_confusions(cm: np.ndarray, names: list[str], k: int = 15) -> list[dict]:
    off = cm.copy()
    np.fill_diagonal(off, 0)
    flat = np.argsort(-off, axis=None)[:k]
    out = []
    for f in flat:
        i, j = np.unravel_index(f, off.shape)
        if off[i, j] == 0:
            break
        out.append({"true": names[i], "pred": names[j], "count": int(off[i, j]), "true_support": int(cm[i].sum())})
    return out
