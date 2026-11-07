"""Detector comparison + selection on VALIDATION (from the official train split).

Combines, per candidate: Ultralytics validation metrics from training (detector_val_metrics.json), our own
pipeline diagnostics at the production threshold (reports/results/pipeline/val_<name>.json: miss rate, IoU
distribution, CPU latency) and the ONNX size. Writes reports/results/detector_comparison.json.

Selection rule (fixed before looking at test data): primary = miss rate at IoU>=0.5 (lower is better);
candidates within 0.5 percentage points are treated as equivalent; among those, prefer the lower CPU latency
(target: free CPU host), then higher mAP50-95.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="directory containing detectors/<name>/")
    ap.add_argument("--names", default="det_yolo26n,det_yolo11n")
    a = ap.parse_args()
    rows, hist = [], {}
    for name in a.names.split(","):
        k = json.loads((Path(a.runs) / "detectors" / name / "detector_val_metrics.json").read_text())
        p = json.loads((ROOT / "reports" / "results" / "pipeline" / f"val_{name}.json").read_text())["detector"]
        rows.append({"name": name, "model": k["model"], "mAP50": k["mAP50"], "mAP50_95": k["mAP50_95"],
                     "precision": k["precision"], "recall": k["recall"], "miss_rate": p["miss_rate"],
                     "no_detection_rate": p["no_detection_rate"], "iou_mean": p["iou_mean"], "iou_median": p["iou_median"],
                     "iou_p10": p["iou_p10"], "images_with_unmatched_dets": p["images_with_unmatched_dets"],
                     "cpu_det_ms_p50": p["cpu_det_ms_p50"], "cpu_det_ms_p95": p["cpu_det_ms_p95"],
                     "onnx_bytes": k["onnx"]["bytes"], "gpu_train": k["env"].get("gpu")})
        hist[name] = p["iou_hist"]
    best_miss = min(r["miss_rate"] for r in rows)
    near = [r for r in rows if r["miss_rate"] <= best_miss + 0.005]
    sel = sorted(near, key=lambda r: (r["cpu_det_ms_p50"], -r["mAP50_95"]))[0]
    out = {"split": "val (from official train)",
           "note": "mAP/P/R from Ultralytics val (conf=0.001); miss rate, IoU and CPU latency from our ONNX pipeline at "
                   "the production threshold (0.25) on this Mac (Apple M1, ONNX Runtime CPU)",
           "runs": rows, "iou_hist": hist, "selected": sel["name"],
           "rationale": f"miss rates within 0.5 pp of the best ({best_miss:.4f}) treated as equivalent: "
                        f"{[r['name'] for r in near]}; lowest CPU p50 latency among them wins"}
    (ROOT / "reports" / "results" / "detector_comparison.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "iou_hist"}, indent=1))


if __name__ == "__main__":
    main()
