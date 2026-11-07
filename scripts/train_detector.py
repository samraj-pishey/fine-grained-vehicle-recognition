"""Fine-tune an Ultralytics YOLO detector (YOLO26n or YOLO11n) on Stanford Cars boxes (single class).

Both candidates use the same data, image size, epochs, batch, optimiser settings and seed. Runs are written
straight to --project (persistent storage); re-running the same command resumes from weights/last.pt.
After training: validation metrics on the validation subset (from the official TRAIN split) + ONNX export.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["yolo26n.pt", "yolo11n.pt"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--fraction", type=float, default=1.0, help="smoke tests only")
    a = ap.parse_args()

    import torch
    import ultralytics
    from ultralytics import YOLO, settings

    settings.update({"sync": False})  # no telemetry
    name = a.name or Path(a.model).stem
    run = Path(a.project) / name
    last = run / "weights" / "last.pt"
    env = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
           "ultralytics": ultralytics.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    print(json.dumps(env))
    done = run / "weights" / "best.pt"
    if last.exists() and not (run / "TRAIN_DONE").exists():
        print("[resume]", last)
        YOLO(str(last)).train(resume=True)
    elif not done.exists():
        YOLO(a.model).train(
            data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, patience=a.patience, seed=a.seed,
            deterministic=True, project=a.project, name=name, exist_ok=True, workers=a.workers, device=a.device,
            single_cls=True, cos_lr=True, plots=True, val=True, fraction=a.fraction, save_period=-1,
        )
    (run / "TRAIN_DONE").write_text("ok")
    best = YOLO(str(done))
    m = best.val(data=a.data, split="val", imgsz=a.imgsz, batch=a.batch, conf=0.001, iou=0.7, plots=True,
                 project=str(run), name="val_eval", exist_ok=True)
    metrics = {"model": a.model, "split": "val (from official train)", "mAP50": float(m.box.map50),
               "mAP50_95": float(m.box.map), "precision": float(m.box.mp), "recall": float(m.box.mr),
               "speed_ms": m.speed, "env": env, "best_pt_sha256": sha256(done), "best_pt_bytes": done.stat().st_size}
    onnx_path = Path(best.export(format="onnx", imgsz=a.imgsz, simplify=True, dynamic=False))
    metrics["onnx"] = {"path": onnx_path.name, "sha256": sha256(onnx_path), "bytes": onnx_path.stat().st_size}
    (run / "detector_val_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
