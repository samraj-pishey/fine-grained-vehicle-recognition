"""Assemble the production deployment bundle from validated artefacts and write a checksummed manifest.

Inputs : selected detector ONNX, selected classifier ONNX (fp32 or int8), artifacts/decision.json (frozen T,
         alpha, threshold fitted on validation), optional classical model (only if mode == hybrid or --with-classical)
Output : artifacts/deployment/{manifest.json, detector.onnx, classifier.onnx, class_mapping.json[, classical_svm.joblib]}
The manifest is validated with the same loader the app uses before the script exits.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vehicle_recognition.inference.manifest import load_manifest, sha256_file  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True)
    ap.add_argument("--detector-arch", required=True)
    ap.add_argument("--det-conf", type=float, default=0.25)
    ap.add_argument("--det-iou", type=float, default=0.7)
    ap.add_argument("--classifier", required=True)
    ap.add_argument("--classifier-arch", required=True)
    ap.add_argument("--img-size", type=int, required=True)
    ap.add_argument("--precision", choices=["fp32", "int8"], default="fp32")
    ap.add_argument("--decision", default=str(ROOT / "artifacts" / "decision.json"))
    ap.add_argument("--classical", default=str(ROOT / "artifacts" / "classical" / "classical_svm.joblib"))
    ap.add_argument("--with-classical", action="store_true", help="ship the classical model for research views")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "deployment"))
    ap.add_argument("--evidence", default="{}", help="JSON string of validation evidence to embed")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    d = json.loads(Path(a.decision).read_text())
    shutil.copy2(a.detector, out / "detector.onnx")
    shutil.copy2(a.classifier, out / "classifier.onnx")
    shutil.copy2(ROOT / "reports" / "dataset" / "class_mapping.json", out / "class_mapping.json")
    classical = None
    if d["mode"] == "hybrid" or a.with_classical:
        shutil.copy2(a.classical, out / "classical_svm.joblib")
        classical = {"path": "classical_svm.joblib", "sha256": sha256_file(out / "classical_svm.joblib"),
                     "bytes": (out / "classical_svm.joblib").stat().st_size, "feature_config_version": 1}
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    m = {
        "schema_version": 1,
        "model_bundle_id": f"vr-{a.detector_arch}-{a.classifier_arch}-{a.img_size}-{a.precision}-{commit}",
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_commit": commit,
        "detector": {"architecture": a.detector_arch, "format": "onnx", "path": "detector.onnx",
                     "sha256": sha256_file(out / "detector.onnx"), "bytes": (out / "detector.onnx").stat().st_size,
                     "input_size": 640, "confidence_threshold": a.det_conf, "nms_iou_threshold": a.det_iou,
                     "min_box_frac": 0.0, "max_detections": 20},
        "classifier": {"architecture": a.classifier_arch, "format": "onnx", "path": "classifier.onnx",
                       "sha256": sha256_file(out / "classifier.onnx"), "bytes": (out / "classifier.onnx").stat().st_size,
                       "input_size": a.img_size, "roi_pad_frac": 0.05, "class_count": 196,
                       "class_mapping": "class_mapping.json", "class_mapping_sha256": sha256_file(out / "class_mapping.json"),
                       "preprocessing": "squash_resize_imagenet_norm", "precision": a.precision},
        "classical": classical,
        "decision": {"mode": d["mode"], "temperature": d["temperature"], "reject_threshold": d["reject_threshold"],
                     "fusion_alpha": d["fusion_alpha"], "classical_temperature": d["classical_temperature"],
                     "threshold_policy": d["threshold_policy"]},
        "evidence": json.loads(a.evidence) | {"decision_fitted_on": d["fitted_on"]},
    }
    (out / "manifest.json").write_text(json.dumps(m, indent=1))
    load_manifest(out / "manifest.json")  # same validation as the app (hashes, schema, mapping)
    print(json.dumps({k: m[k] for k in ("model_bundle_id", "decision")}, indent=1))
    print("manifest sha256:", sha256_file(out / "manifest.json"))
    for f in sorted(out.iterdir()):
        print(f"  {f.name:24s} {f.stat().st_size / 1e6:8.2f} MB")


if __name__ == "__main__":
    main()
