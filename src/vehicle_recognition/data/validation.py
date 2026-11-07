"""Dataset integrity validation: decoding, image properties, box validity, duplicates, distribution.

Memory-aware: images are decoded one at a time in a small thread pool; nothing is held in RAM beyond
per-image scalars and 64-bit perceptual hashes.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}


@dataclass
class ImageFacts:
    path: str
    exists: bool
    bytes: int = -1
    sha256: str = ""
    decodes: bool = False
    error: str = ""
    format: str = ""
    mode: str = ""
    width: int = -1
    height: int = -1
    exif_orientation: int = 1
    phash: str = ""


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def inspect_image(path: Path) -> ImageFacts:
    import imagehash

    facts = ImageFacts(path=str(path), exists=path.is_file())
    if not facts.exists:
        return facts
    facts.bytes = path.stat().st_size
    facts.sha256 = _sha256(path)
    try:
        with Image.open(path) as im:
            im.verify()  # structural check
        with Image.open(path) as im:
            facts.format = im.format or ""
            facts.mode = im.mode
            facts.width, facts.height = im.size  # stored (pre-EXIF) size; annotations refer to this
            try:
                facts.exif_orientation = int(im.getexif().get(0x0112, 1))
            except Exception:
                facts.exif_orientation = 1
            im.load()  # full decode, catches truncation
            facts.phash = str(imagehash.phash(im.convert("L")))
            facts.decodes = True
    except Exception as e:  # noqa: BLE001 - we record every failure reason
        facts.error = f"{type(e).__name__}: {e}"[:300]
    return facts


def inspect_images(paths: list[Path], workers: int = 4) -> pd.DataFrame:
    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(inspect_image, paths, chunksize=32))
    return pd.DataFrame([r.__dict__ for r in rows])


def discover_structure(root: Path, max_tree_entries: int = 60) -> dict:
    """Recursive structure discovery (no decoding)."""
    ext_counts: Counter = Counter()
    dir_counts: Counter = Counter()
    total = 0
    names_by_dir: dict[str, list[str]] = {}
    for p in root.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
            ext_counts[p.suffix.lower()] += 1
            rel_dir = str(p.parent.relative_to(root))
            dir_counts[rel_dir] += 1
            if p.suffix.lower() in IMG_EXTS:
                names_by_dir.setdefault(rel_dir, []).append(p.name)
    # duplicate filenames across directories
    seen: dict[str, list[str]] = {}
    for d, names in names_by_dir.items():
        for n in names:
            seen.setdefault(n, []).append(d)
    dup_names = {n: ds for n, ds in seen.items() if len(ds) > 1}
    tree = sorted(dir_counts.items())[:max_tree_entries]
    return {
        "root": str(root),
        "total_bytes": total,
        "files_by_extension": dict(ext_counts),
        "files_per_directory": dict(tree),
        "filenames_in_multiple_dirs": len(dup_names),
        "filenames_in_multiple_dirs_example": dict(list(dup_names.items())[:5]),
    }


def box_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Adds validity flags. Requires width/height columns (stored image size)."""
    out = df.copy()
    out["box_w"] = out["x_max"] - out["x_min"]
    out["box_h"] = out["y_max"] - out["y_min"]
    out["box_positive"] = (out["box_w"] > 0) & (out["box_h"] > 0)
    out["box_in_bounds"] = (
        (out["x_min"] >= 0) & (out["y_min"] >= 0) & (out["x_max"] <= out["width"]) & (out["y_max"] <= out["height"])
    )
    out["box_area_frac"] = (out["box_w"] * out["box_h"]) / (out["width"] * out["height"]).clip(lower=1)
    out["annotation_valid"] = out["box_positive"] & out["box_in_bounds"] & out["decodes"]
    return out


def coordinate_convention_evidence(df: pd.DataFrame) -> dict:
    """Empirical evidence for 1-based inclusive MATLAB indexing of the original box values."""
    w, h = df["width"], df["height"]
    return {
        "n": int(len(df)),
        "orig_x1_eq_0": int((df["orig_x1"] == 0).sum()),
        "orig_y1_eq_0": int((df["orig_y1"] == 0).sum()),
        "orig_x1_eq_1": int((df["orig_x1"] == 1).sum()),
        "orig_y1_eq_1": int((df["orig_y1"] == 1).sum()),
        "orig_x2_eq_width": int((df["orig_x2"] == w).sum()),
        "orig_y2_eq_height": int((df["orig_y2"] == h).sum()),
        "orig_x2_gt_width": int((df["orig_x2"] > w).sum()),
        "orig_y2_gt_height": int((df["orig_y2"] > h).sum()),
        "orig_x2_eq_width_minus_1": int((df["orig_x2"] == w - 1).sum()),
    }


def hamming_hex(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def near_duplicate_candidates(
    df: pd.DataFrame, max_dist: int = 4, cross_split_only: bool = False
) -> list[tuple[int, int, int]]:
    """pHash candidates via exact bucket on 4 x 16-bit bands (pigeonhole: any pair with distance <= 3
    shares a band; we then verify distance <= max_dist). Returns (idx_a, idx_b, distance)."""
    hashes = df["phash"].tolist()
    splits = df["split"].tolist()
    ints = [int(h, 16) if h else None for h in hashes]
    buckets: dict[tuple[int, int], list[int]] = {}
    for i, v in enumerate(ints):
        if v is None:
            continue
        for band in range(4):
            buckets.setdefault((band, (v >> (16 * band)) & 0xFFFF), []).append(i)
    pairs: dict[tuple[int, int], int] = {}
    for members in buckets.values():
        if len(members) < 2 or len(members) > 2000:
            continue
        for ai in range(len(members)):
            for bi in range(ai + 1, len(members)):
                a, b = members[ai], members[bi]
                if cross_split_only and splits[a] == splits[b]:
                    continue
                key = (a, b) if a < b else (b, a)
                if key in pairs:
                    continue
                d = bin(ints[a] ^ ints[b]).count("1")
                if d <= max_dist:
                    pairs[key] = d
    return [(a, b, d) for (a, b), d in pairs.items()]


def ssim_pair(path_a: str, path_b: str, size: int = 256) -> float:
    from skimage.metrics import structural_similarity

    def load(p):
        with Image.open(p) as im:
            return np.asarray(ImageOps.exif_transpose(im).convert("L").resize((size, size)), dtype=np.float32)

    return float(structural_similarity(load(path_a), load(path_b), data_range=255.0))


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
