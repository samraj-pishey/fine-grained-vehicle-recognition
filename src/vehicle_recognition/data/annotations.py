"""Stanford Cars annotation parsing (original MATLAB devkit format) and class mapping.

Coordinate convention (verified by `scripts/validate_dataset.py`, see dataset_validation_report.md):
the devkit stores bbox_x1/y1/x2/y2 as MATLAB pixel indices (1-based, inclusive). The canonical table
converts them to 0-based, half-open float pixel coordinates:

    x_min = x1 - 1,  y_min = y1 - 1,  x_max = x2,  y_max = y2

so that width = x2 - x1 + 1 and a box covering the whole image is [0, 0, W, H]. Original values are
retained in `orig_x1 .. orig_y2`. Class IDs are converted from 1-based (MATLAB) to 0-based `class_id`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

NUM_CLASSES = 196
# Makes spanning more than one token in the official class names. Verified against all 196 names
# (tests/test_annotations.py::test_all_official_names_parse).
MULTI_WORD_MAKES = ("AM General", "Aston Martin", "Land Rover")
_YEAR_RE = re.compile(r"^(19|20)\d{2}$")


@dataclass(frozen=True)
class ClassInfo:
    class_id: int  # 0-based canonical id
    source_class: int  # 1-based id in the devkit
    name: str  # official class name, verbatim
    make: str
    model: str  # model + body/trim text as given in the official name
    year: int

    def to_dict(self) -> dict:
        return asdict(self)


def parse_class_name(name: str) -> tuple[str, str, int]:
    """Split an official name 'Make Model... Year' into (make, model, year). Raises on malformed names."""
    tokens = name.strip().split()
    if len(tokens) < 3 or not _YEAR_RE.match(tokens[-1]):
        raise ValueError(f"Unparseable class name (no trailing year): {name!r}")
    year = int(tokens[-1])
    body = " ".join(tokens[:-1])
    for mk in MULTI_WORD_MAKES:
        if body.startswith(mk + " "):
            return mk, body[len(mk) + 1 :], year
    return tokens[0], " ".join(tokens[1:-1]), year


def _scalar(x):
    """Unwrap scipy.io MATLAB cell/struct leaves (arrays of shape (1,1) or (1,))."""
    a = np.asarray(x)
    while a.dtype == object and a.size == 1:
        a = np.asarray(a.item())
    if a.dtype.kind in "US":
        return str(a.reshape(-1)[0]) if a.size else ""
    return a.reshape(-1)[0].item()


def load_class_names(meta_mat: Path) -> list[str]:
    from scipy.io import loadmat

    m = loadmat(str(meta_mat))
    names = [_scalar(c) for c in m["class_names"].reshape(-1)]
    if len(names) != NUM_CLASSES:
        raise ValueError(f"Expected {NUM_CLASSES} class names, found {len(names)}")
    return names


def build_class_mapping(names: Iterable[str]) -> list[ClassInfo]:
    out = []
    for i, n in enumerate(names):
        make, model, year = parse_class_name(n)
        out.append(ClassInfo(class_id=i, source_class=i + 1, name=n, make=make, model=model, year=year))
    if len({c.name for c in out}) != len(out):
        raise ValueError("Duplicate class names in metadata")
    return out


def load_annotation_mat(path: Path, split: str) -> pd.DataFrame:
    """Parse a devkit annotation struct array into a DataFrame with original + canonical columns."""
    from scipy.io import loadmat

    ann = loadmat(str(path))["annotations"].reshape(-1)
    fields = ann.dtype.names
    rows = []
    for idx, rec in enumerate(ann):
        r = {f: _scalar(rec[f]) for f in fields}
        rows.append(
            {
                "split": split,
                "source_index": idx + 1,  # 1-based position in the .mat (devkit submission order)
                "fname": r["fname"],
                "orig_x1": int(r["bbox_x1"]),
                "orig_y1": int(r["bbox_y1"]),
                "orig_x2": int(r["bbox_x2"]),
                "orig_y2": int(r["bbox_y2"]),
                "source_class": int(r["class"]) if "class" in fields else -1,
            }
        )
    df = pd.DataFrame(rows)
    df["x_min"] = df["orig_x1"].astype(float) - 1.0
    df["y_min"] = df["orig_y1"].astype(float) - 1.0
    df["x_max"] = df["orig_x2"].astype(float)
    df["y_max"] = df["orig_y2"].astype(float)
    df["class_id"] = np.where(df["source_class"] > 0, df["source_class"] - 1, -1)
    return df


def matlab_box_to_canonical(x1: int, y1: int, x2: int, y2: int) -> tuple[float, float, float, float]:
    return float(x1 - 1), float(y1 - 1), float(x2), float(y2)


def canonical_box_to_matlab(x_min: float, y_min: float, x_max: float, y_max: float) -> tuple[int, int, int, int]:
    return int(round(x_min + 1)), int(round(y_min + 1)), int(round(x_max)), int(round(y_max))


def build_canonical_table(devkit: Path, test_labels_mat: Path) -> pd.DataFrame:
    names = load_class_names(devkit / "cars_meta.mat")
    mapping = build_class_mapping(names)
    tr = load_annotation_mat(devkit / "cars_train_annos.mat", "train")
    te = load_annotation_mat(test_labels_mat, "test")
    # Cross-check: the labelled test file must agree with the unlabelled devkit test file on boxes/fnames.
    te_nolab = load_annotation_mat(devkit / "cars_test_annos.mat", "test")
    cols = ["fname", "orig_x1", "orig_y1", "orig_x2", "orig_y2"]
    if not te[cols].reset_index(drop=True).equals(te_nolab[cols].reset_index(drop=True)):
        raise ValueError("cars_test_annos_withlabels.mat disagrees with devkit cars_test_annos.mat")
    df = pd.concat([tr, te], ignore_index=True)
    df["relative_path"] = np.where(df["split"] == "train", "cars_train/", "cars_test/") + df["fname"]
    df["class_name"] = df["class_id"].map({c.class_id: c.name for c in mapping})
    df["source_annotation_id"] = df["split"] + ":" + df["source_index"].astype(str)
    return df
