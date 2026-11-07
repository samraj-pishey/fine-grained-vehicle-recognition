"""Build training inputs from extracted images + annotation table.

  --task crops : classifier ROI crop cache + crops_index.csv
  --task yolo  : Ultralytics detection dataset (single class 'vehicle'): images/ (symlinks) + labels/ + data.yaml
  --task package : (Mac) write the cloud package annotation table with TRAIN/VAL rows only

Test rows are refused unless --allow-test is given (used only by the frozen final evaluation on the Mac).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vehicle_recognition.classification.data import CropPolicy, build_crop_cache  # noqa: E402

COLS = ["relative_path", "split", "subset", "class_id", "x_min", "y_min", "x_max", "y_max", "width", "height", "sha256"]


def load_table(path: Path, allow_test: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not allow_test and (df["split"] == "test").any():
        raise SystemExit("Refusing: table contains official test rows (use --allow-test only for final evaluation)")
    return df


def task_package(a):
    canon = pd.read_csv(a.canonical)
    tv = canon[(canon["split"] == "train") & canon["subset"].isin(["train", "val"])][COLS]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tv.to_csv(out / "train_val_annotations.csv", index=False)
    print("wrote", out / "train_val_annotations.csv", tv["subset"].value_counts().to_dict())


def task_crops(a):
    df = load_table(Path(a.table), a.allow_test)
    subsets = a.subsets.split(",")
    df = df[df["subset"].isin(subsets)]
    out = Path(a.out)
    policy = CropPolicy(roi_pad_frac=a.roi_pad_frac)
    parts = []
    n = len(df)
    for i in range(0, n, 500):
        parts.append(build_crop_cache(df.iloc[i : i + 500], Path(a.image_root), out, policy))
        print(f"  crops {min(n, i + 500)}/{n}", flush=True)
    idx = pd.concat(parts, ignore_index=True)
    name = a.index_name or ("crops_index.csv" if not a.allow_test else "crops_index_test.csv")
    idx.to_csv(out / name, index=False)
    print("wrote", out / name, idx["subset"].value_counts().to_dict())


def task_yolo(a):
    df = load_table(Path(a.table), a.allow_test)
    out = Path(a.out)
    subsets = a.subsets.split(",")
    for sub in subsets:
        (out / "images" / sub).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sub).mkdir(parents=True, exist_ok=True)
    n_written = {s: 0 for s in subsets}
    n_exif_fixed = 0
    for r in df[df["subset"].isin(subsets)].itertuples(index=False):
        src = Path(a.image_root) / r.relative_path
        w, h = int(r.width), int(r.height)
        if w <= 0 or h <= 0:
            with Image.open(src) as im:
                w, h = im.size
        # canonical boxes are 0-based half-open; clip, then normalised cx, cy, w, h
        x0, y0 = max(0.0, r.x_min), max(0.0, r.y_min)
        x1, y1 = min(float(w), r.x_max), min(float(h), r.y_max)
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        stem = Path(r.relative_path).stem
        dst = out / "images" / r.subset / f"{stem}.jpg"
        if not dst.exists() and not dst.with_suffix(".png").exists():
            # Annotations refer to STORED pixels. OpenCV/Ultralytics apply EXIF orientation when reading, so any
            # image with a rotating EXIF tag (Stanford Cars: cars_train/07389.jpg has a wrong tag 6) is written as
            # an EXIF-free lossless PNG copy of the stored pixels; everything else is a symlink to the raw file.
            with Image.open(src) as im:
                orient = int(im.getexif().get(0x0112, 1) or 1)
                if orient != 1:
                    im.convert("RGB").save(dst.with_suffix(".png"))
                    n_exif_fixed += 1
                else:
                    os.symlink(src.resolve(), dst)
        (out / "labels" / r.subset / f"{stem}.txt").write_text(
            f"0 {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}\n")
        n_written[r.subset] += 1
    lines = [f"path: {out.resolve()}"] + [f"{s}: images/{s}" for s in subsets] + ["names:", "  0: vehicle"]
    (out / "data.yaml").write_text("\n".join(lines) + "\n")
    print("yolo dataset", n_written, "exif-stripped copies:", n_exif_fixed, "->", out / "data.yaml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["package", "crops", "yolo"], required=True)
    ap.add_argument("--canonical", help="canonical_annotations.csv (package task)")
    ap.add_argument("--table", help="annotation table (train_val_annotations.csv, or canonical for final eval)")
    ap.add_argument("--image-root", help="directory containing cars_train/ (and cars_test/ for final eval)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--subsets", default="train,val")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--roi-pad-frac", type=float, default=0.05)
    ap.add_argument("--index-name", default=None)
    a = ap.parse_args()
    {"package": task_package, "crops": task_crops, "yolo": task_yolo}[a.task](a)


if __name__ == "__main__":
    main()
