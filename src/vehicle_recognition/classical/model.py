"""Classical classifier: family-wise scaling -> optional PCA -> RBF-SVM with probability calibration.

Probability protocol (documented choice): `CalibratedClassifierCV(SVC(), method="sigmoid", cv=5,
ensemble=False)` - out-of-fold one-vs-rest SVM decision values from a stratified 5-fold split of the TRAINING
data are used to fit per-class Platt sigmoids; the final SVM is refit on all training data and the outputs are
normalised over classes. (This is scikit-learn's documented replacement for the deprecated
`SVC(probability=True)`.) Validation data are used only for model selection, evaluation and (later) a single
temperature parameter; test data never enter any fit. Output columns are re-indexed to canonical order 0..195.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .features import FAMILIES


class FamilyScaler(BaseEstimator, TransformerMixin):
    """Standardise each feature family independently, then weight each family by 1/sqrt(dim) so that every
    family contributes equal expected squared norm to the RBF kernel distance regardless of its size."""

    def __init__(self, family_dims: tuple[int, ...] = (), weighting: str = "equal"):
        self.family_dims = family_dims
        self.weighting = weighting  # "equal": x/sqrt(dim) per family; "none": plain per-feature standardisation

    def fit(self, X, y=None):
        self.scalers_ = []
        start = 0
        for d in self.family_dims:
            self.scalers_.append(StandardScaler().fit(X[:, start : start + d]))
            start += d
        if start != X.shape[1]:
            raise ValueError(f"family_dims sum {start} != n_features {X.shape[1]}")
        return self

    def transform(self, X):
        out, start = [], 0
        for d, sc in zip(self.family_dims, self.scalers_, strict=True):
            z = sc.transform(X[:, start : start + d])
            out.append(z / np.sqrt(d) if self.weighting == "equal" else z)
            start += d
        return np.hstack(out).astype(np.float32)


@dataclass
class ClassicalSpec:
    families: tuple[str, ...] = FAMILIES
    pca_components: int | float | None = None  # None = no PCA; float in (0,1) = variance fraction
    family_weighting: str = "equal"
    C: float = 10.0
    gamma: str | float = "scale"
    class_weight: str | None = None
    calibration: str = "sigmoid"  # Platt; "isotonic" needs far more data per class than Stanford Cars has
    calibration_cv: int = 5
    seed: int = 0
    extra: dict = field(default_factory=dict)


def stack_families(feats: dict[str, np.ndarray], families) -> tuple[np.ndarray, tuple[int, ...]]:
    mats = [np.asarray(feats[f], dtype=np.float32) for f in families]
    return np.hstack(mats), tuple(m.shape[1] for m in mats)


def build_pipeline(spec: ClassicalSpec, family_dims: tuple[int, ...]) -> Pipeline:
    steps = [("scale", FamilyScaler(family_dims, getattr(spec, "family_weighting", "equal")))]
    if spec.pca_components is not None:
        steps.append(("pca", PCA(n_components=spec.pca_components, svd_solver="full", random_state=spec.seed)))
    svc = SVC(kernel="rbf", C=spec.C, gamma=spec.gamma, class_weight=spec.class_weight,
              decision_function_shape="ovr", random_state=spec.seed, cache_size=1000)
    steps.append(("svm", CalibratedClassifierCV(svc, method=spec.calibration, cv=spec.calibration_cv, ensemble=False)))
    return Pipeline(steps)


def predict_proba_canonical(pipe: Pipeline, X: np.ndarray, num_classes: int = 196) -> np.ndarray:
    """Probabilities with columns in canonical class order (missing classes -> 0), rows renormalised."""
    p = pipe.predict_proba(X)
    out = np.zeros((X.shape[0], num_classes), dtype=np.float64)
    out[:, pipe.classes_.astype(int)] = p
    s = out.sum(axis=1, keepdims=True)
    out = out / np.where(s > 0, s, 1.0)
    return out


def save_model(path, pipe: Pipeline, spec: ClassicalSpec, feature_cfg: dict, family_dims, meta: dict) -> None:
    joblib.dump({"pipeline": pipe, "spec": spec.__dict__, "feature_config": feature_cfg,
                 "family_dims": family_dims, "meta": meta}, path, compress=3)


def load_model(path) -> dict:
    return joblib.load(path)
