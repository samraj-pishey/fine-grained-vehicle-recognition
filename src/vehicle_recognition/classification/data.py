"""Classifier datasets built on a regenerable ROI crop cache.

Crop cache: each image's ground-truth box is cropped with generous context (`cache_pad_frac`, default 0.15),
downscaled so the longest side <= `cache_max_side`, saved as JPEG (derived data only), and the box location
inside the cached crop is recorded. This lets
  * evaluation reproduce exactly the inference-time crop (box + `roi_pad_frac`) and
  * training apply box jitter (simulating detector localisation noise) using real surrounding context.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ..preprocessing.roi import crop_roi
from .preprocess import IMAGENET_MEAN, IMAGENET_STD, resize_square


@dataclass(frozen=True)
class CropPolicy:
    roi_pad_frac: float = 0.05  # MUST equal the inference-time padding applied to detector boxes
    cache_pad_frac: float = 0.15
    cache_max_side: int = 512


def build_crop_cache(rows: pd.DataFrame, image_root: Path, out_dir: Path, policy: CropPolicy) -> pd.DataFrame:
    """rows need: relative_path, subset, class_id, x_min..y_max. Returns an index DataFrame."""
    out_dir.mkdir(parents=True, exist_ok=True)
    recs = []
    for r in rows.itertuples(index=False):
        dst = out_dir / r.subset / Path(r.relative_path).name
        with Image.open(image_root / r.relative_path) as im:
            img = np.asarray(im.convert("RGB"))
        h, w = img.shape[:2]
        box = (max(0.0, r.x_min), max(0.0, r.y_min), min(float(w), r.x_max), min(float(h), r.y_max))
        bw, bh = box[2] - box[0], box[3] - box[1]
        ox = max(0, int(np.floor(box[0] - bw * policy.cache_pad_frac)))
        oy = max(0, int(np.floor(box[1] - bh * policy.cache_pad_frac)))
        crop = crop_roi(img, box, pad_frac=policy.cache_pad_frac)
        s = min(1.0, policy.cache_max_side / max(crop.shape[:2]))
        if s < 1.0:
            crop = np.asarray(Image.fromarray(crop).resize(
                (max(1, round(crop.shape[1] * s)), max(1, round(crop.shape[0] * s))), Image.BICUBIC))
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            Image.fromarray(crop).save(dst, quality=95)
        recs.append({
            "subset": r.subset, "relative_path": r.relative_path, "class_id": int(r.class_id),
            "cache_path": str(dst.relative_to(out_dir)), "scale": s,
            "bx0": (box[0] - ox) * s, "by0": (box[1] - oy) * s, "bx1": (box[2] - ox) * s, "by1": (box[3] - oy) * s,
        })
    return pd.DataFrame(recs)


class CropDataset:
    """torch-style dataset (no torch import at module level so the runtime image stays torch-free)."""

    def __init__(self, index: pd.DataFrame, cache_root: Path, img_size: int, policy: CropPolicy,
                 train: bool = False, aug: dict | None = None, seed: int = 0):
        self.df = index.reset_index(drop=True)
        self.root, self.size, self.policy, self.train = Path(cache_root), img_size, policy, train
        self.aug = aug or {}
        self.tf = build_train_tf(self.aug) if train else None
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.df)

    def _box(self, r):
        box = np.array([r.bx0, r.by0, r.bx1, r.by1], dtype=np.float64)
        if self.train and self.aug.get("box_jitter", 0) > 0:
            j = self.aug["box_jitter"]
            w, h = box[2] - box[0], box[3] - box[1]
            cx, cy = (box[0] + box[2]) / 2 + np.random.uniform(-j, j) * w, (box[1] + box[3]) / 2 + np.random.uniform(-j, j) * h
            w, h = w * np.random.uniform(1 - j, 1 + j), h * np.random.uniform(1 - j, 1 + j)
            box = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        return box

    def __getitem__(self, i):
        import torch

        r = self.df.iloc[i]
        with Image.open(self.root / r.cache_path) as im:
            img = np.asarray(im.convert("RGB"))
        roi = crop_roi(img, self._box(r), pad_frac=self.policy.roi_pad_frac)
        sq = resize_square(roi, self.size)
        if self.train:
            x = self.tf(torch.from_numpy(sq).permute(2, 0, 1))  # uint8 CHW -> float normalised
        else:
            x = torch.from_numpy(((sq.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).copy())
        return x, int(r.class_id)


def build_train_tf(aug: dict):
    import torch
    from torchvision.transforms import v2

    t = [v2.RandomHorizontalFlip(0.5)] if aug.get("hflip", True) else []
    if aug.get("rotation", 0):
        t.append(v2.RandomRotation(aug["rotation"]))
    if aug.get("perspective", 0):
        t.append(v2.RandomPerspective(distortion_scale=aug["perspective"], p=0.3))
    if aug.get("color_jitter"):
        b, c, s = aug["color_jitter"]
        t.append(v2.ColorJitter(brightness=b, contrast=c, saturation=s))
    if aug.get("blur_p", 0):
        t.append(v2.RandomApply([v2.GaussianBlur(5, sigma=(0.1, 1.5))], p=aug["blur_p"]))
    t += [v2.ToDtype(torch.float32, scale=True), v2.Normalize(IMAGENET_MEAN.tolist(), IMAGENET_STD.tolist())]
    if aug.get("erase_p", 0):
        t.append(v2.RandomErasing(p=aug["erase_p"], scale=(0.02, 0.15)))
    return v2.Compose(t)
