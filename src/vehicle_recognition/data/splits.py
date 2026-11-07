"""Deterministic, grouped, class-stratified train/validation split drawn from the OFFICIAL TRAIN split only.

Duplicate clusters (exact SHA-256 or confirmed near-duplicates) are kept on the same side of the split so
that validation cannot leak through duplicates. The official test split is never touched here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class _UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def duplicate_groups(n: int, pairs: list[tuple[int, int]]) -> np.ndarray:
    uf = _UnionFind(n)
    for a, b in pairs:
        uf.union(a, b)
    return np.array([uf.find(i) for i in range(n)])


def make_train_val_split(
    train_df: pd.DataFrame, val_fraction: float, seed: int, group_col: str = "dup_group"
) -> pd.Series:
    """Return a Series (index aligned with train_df) of 'train' / 'val'.

    For each class, whole groups are shuffled and assigned to validation until the class's validation
    quota round(val_fraction * n_class) (min 1) is reached. Groups spanning several classes are assigned
    by their first member's class.
    """
    if (train_df["split"] != "train").any():
        raise ValueError("Validation split may only be drawn from the official train split")
    rng = np.random.default_rng(seed)
    assignment = pd.Series("train", index=train_df.index, dtype=object)
    groups = train_df[group_col] if group_col in train_df else pd.Series(np.arange(len(train_df)), index=train_df.index)
    first_class = train_df.groupby(groups)["class_id"].transform("first")
    for cls in sorted(train_df["class_id"].unique()):
        idx = train_df.index[first_class == cls]
        g_ids = groups.loc[idx].unique()
        g_ids = g_ids[rng.permutation(len(g_ids))]
        quota = max(1, int(round(val_fraction * len(idx))))
        taken = 0
        for g in g_ids:
            if taken >= quota:
                break
            members = idx[groups.loc[idx] == g]
            assignment.loc[members] = "val"
            taken += len(members)
    return assignment
