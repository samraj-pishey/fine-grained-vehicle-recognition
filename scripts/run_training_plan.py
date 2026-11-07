"""Execute the GPU experiment plan (configs/training_plan.yaml) end to end, resumably.

Typical cloud usage (see notebooks/):
  python scripts/run_training_plan.py --archive <cars_train.tgz> --table <train_val_annotations.csv> \
      --work /content/work --out /content/drive/MyDrive/vehicle_recognition/runs [--only det_yolo26n,cls_mnv3l_224]

--work : fast ephemeral disk (extracted images, crop cache, YOLO dataset) - regenerated if lost
--out  : persistent storage (Drive / Kaggle working dir) - checkpoints, logs, metrics
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def sh(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=REPO)
    if r.returncode != 0:
        raise SystemExit(f"command failed ({r.returncode}): {' '.join(cmd)}")


def prepare(a, work: Path) -> tuple[Path, Path]:
    src_cfg = yaml.safe_load((REPO / "configs" / "dataset_sources.yaml").read_text())
    expected = src_cfg["files"]["cars_train.tgz"]["sha256"]
    marker = work / "PREPARED"
    if marker.exists():
        print("[prepare] already done")
        return work / "crops", work / "yolo"
    work.mkdir(parents=True, exist_ok=True)
    t = time.time()
    got = sha256(Path(a.archive))
    if got != expected:
        raise SystemExit(f"cars_train.tgz SHA-256 mismatch: {got} != {expected}")
    print(f"[prepare] archive verified ({time.time() - t:.0f}s); extracting")
    with tarfile.open(a.archive, "r:gz") as tf:
        tf.extractall(work / "images", filter="data")
    shutil.copy(a.table, work / "train_val_annotations.csv")
    sh([PY, "scripts/prepare_training_data.py", "--task", "crops", "--table", str(work / "train_val_annotations.csv"),
        "--image-root", str(work / "images"), "--out", str(work / "crops")])
    sh([PY, "scripts/prepare_training_data.py", "--task", "yolo", "--table", str(work / "train_val_annotations.csv"),
        "--image-root", str(work / "images"), "--out", str(work / "yolo")])
    marker.write_text("ok")
    return work / "crops", work / "yolo"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="", help="comma-separated run names (default: all)")
    ap.add_argument("--skip-detectors", action="store_true")
    ap.add_argument("--skip-classifiers", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    plan = yaml.safe_load((REPO / "configs" / "training_plan.yaml").read_text())
    only = {s for s in a.only.split(",") if s}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    crops, yolo = prepare(a, Path(a.work))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    (out / "plan_snapshot.json").write_text(json.dumps({"commit": commit, "plan": plan}, indent=2))

    if not a.skip_detectors:
        c = plan["detector"]["common"]
        for r in plan["detector"]["runs"]:
            if only and r["name"] not in only:
                continue
            if (out / "detectors" / r["name"] / "detector_val_metrics.json").exists():
                print("[skip]", r["name"])
                continue
            sh([PY, "scripts/train_detector.py", "--model", r["model"], "--data", str(yolo / "data.yaml"),
                "--project", str(out / "detectors"), "--name", r["name"], "--epochs", str(c["epochs"]),
                "--imgsz", str(c["imgsz"]), "--batch", str(c["batch"]), "--patience", str(c["patience"]),
                "--seed", str(c["seed"]), "--workers", str(a.workers)])

    if not a.skip_classifiers:
        c = plan["classifier"]["common"]
        for r in plan["classifier"]["runs"]:
            if only and r["name"] not in only:
                continue
            cfg = {**c, **r}
            run_dir = out / "classifiers" / r["name"]
            if (run_dir / "DONE").exists():
                print("[skip]", r["name"])
                continue
            cmd = [PY, "scripts/train_classifier.py", "--arch", cfg["arch"], "--img-size", str(cfg["img_size"]),
                   "--crop-index", str(crops / "crops_index.csv"), "--cache-root", str(crops), "--out", str(run_dir),
                   "--epochs", str(cfg["epochs"]), "--freeze-epochs", str(cfg["freeze_epochs"]),
                   "--batch-size", str(cfg["batch_size"]), "--lr", str(cfg["lr"]),
                   "--backbone-lr-mult", str(cfg["backbone_lr_mult"]), "--weight-decay", str(cfg["weight_decay"]),
                   "--label-smoothing", str(cfg["label_smoothing"]), "--aug", cfg["aug"], "--sampler", cfg["sampler"],
                   "--patience", str(cfg["patience"]), "--seed", str(cfg["seed"]), "--workers", str(a.workers)]
            if cfg.get("amp"):
                cmd.append("--amp")
            if cfg.get("class_weighted_loss"):
                cmd.append("--class-weighted-loss")
            sh(cmd)
    print("\nPLAN COMPLETE. Outputs in", out)


if __name__ == "__main__":
    main()
