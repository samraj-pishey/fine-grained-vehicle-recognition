"""Cloud-side data acquisition (no manual upload needed).

Downloads the ORIGINAL car_devkit.tgz and cars_train.tgz from the pinned Internet Archive captures with parallel
HTTP range requests, verifies SHA-256 against configs/dataset_sources.yaml, then rebuilds
train_val_annotations.csv from the official devkit + the committed split assignment
(configs/split_assignment.csv: filenames and train/val/excluded only - no labels). The official test split is
never downloaded here.

  python scripts/fetch_cloud_data.py --out /content/data
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
import threading
import time
import urllib.request
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
UA = {"User-Agent": "vehicle-recognition-dataset-fetch/1.0"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def _size(url: str) -> int:
    for attempt in range(20):
        try:
            req = urllib.request.Request(url, method="HEAD", headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                return int(r.headers["Content-Length"])
        except Exception as e:  # noqa: BLE001 - the archive rate-limits; back off and retry
            print(f"  size request retry {attempt + 1}: {type(e).__name__}", flush=True)
            time.sleep(min(60, 5 * (attempt + 1)))
    raise RuntimeError(f"could not get size of {url}")


def _fetch_range(url: str, start: int, end: int, part: Path) -> None:
    for attempt in range(30):
        have = part.stat().st_size if part.exists() else 0
        if start + have > end:
            return
        try:
            req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={start + have}-{end}"})
            with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab") as f:
                while chunk := r.read(1 << 18):
                    f.write(chunk)
        except Exception as e:  # noqa: BLE001 - network errors are retried from the current offset
            print(f"  retry {attempt + 1} for {part.name}: {type(e).__name__}", flush=True)
            time.sleep(min(60, 5 * (attempt + 1)))
    raise RuntimeError(f"failed to download {part}")


def download(url: str, dst: Path, expected_sha: str, n: int = 12) -> None:
    if dst.exists() and sha256(dst) == expected_sha:
        print(f"[ok] {dst.name} already present and verified")
        return
    total = _size(url)
    chunk = -(-total // n)
    parts = [dst.with_name(f"{dst.name}.p{i}") for i in range(n)]
    threads = [threading.Thread(target=_fetch_range, args=(url, i * chunk, min(total, (i + 1) * chunk) - 1, parts[i]))
               for i in range(n) if i * chunk < total]
    t0 = time.time()
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        got = sum(p.stat().st_size for p in parts if p.exists())
        print(f"  {dst.name}: {got / 1e6:.0f}/{total / 1e6:.0f} MB ({got / max(1, time.time() - t0) / 1e6:.2f} MB/s)", flush=True)
        time.sleep(20)
    with open(dst, "wb") as out:
        for p in parts:
            if p.exists():
                out.write(p.read_bytes())
                p.unlink()
    got = sha256(dst)
    if got != expected_sha:
        dst.unlink()
        raise SystemExit(f"SHA-256 mismatch for {dst.name}: {got}")
    print(f"[ok] {dst.name} {total} bytes, SHA-256 verified ({time.time() - t0:.0f}s)")


def build_table(devkit_tgz: Path, out_dir: Path) -> Path:
    from vehicle_recognition.data.annotations import load_annotation_mat

    with tarfile.open(devkit_tgz, "r:gz") as tf:
        tf.extractall(out_dir, filter="data")
    ann = load_annotation_mat(out_dir / "devkit" / "cars_train_annos.mat", "train")
    ann["relative_path"] = "cars_train/" + ann["fname"]
    split = pd.read_csv(ROOT / "configs" / "split_assignment.csv")
    t = ann.merge(split, on="relative_path", how="inner", validate="one_to_one")
    t = t[t["subset"].isin(["train", "val"])]
    t["width"], t["height"], t["sha256"] = -1, -1, ""
    cols = ["relative_path", "split", "subset", "class_id", "x_min", "y_min", "x_max", "y_max", "width", "height", "sha256"]
    path = out_dir / "train_val_annotations.csv"
    t[cols].to_csv(path, index=False)
    print("[ok] table", path, t["subset"].value_counts().to_dict())
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--connections", type=int, default=12)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    src = yaml.safe_load((ROOT / "configs" / "dataset_sources.yaml").read_text())
    for name in ("car_devkit.tgz", "cars_train.tgz"):
        f = src["files"][name]
        download(src["wayback_prefix"] + f["official_url"], out / name, f["sha256"], a.connections)
    build_table(out / "car_devkit.tgz", out)


if __name__ == "__main__":
    main()
