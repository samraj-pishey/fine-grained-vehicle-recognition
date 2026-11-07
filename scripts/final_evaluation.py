"""FROZEN final evaluation on the official Stanford Cars TEST split (run once).

Preconditions (checked): clean git working tree; deployment bundle + decision.json present; no previous final
result (unless --rerun-disclosed, which is recorded). A freeze record with hashes of every config/artefact is
written BEFORE any test image is read.

Reports (all on the official test split, 8,041 images):
  * deep / classical / hybrid on ground-truth ROIs: top-1/5, macro-F1, ECE/NLL/Brier
  * selective prediction with the frozen threshold: coverage, selective accuracy
  * detector + three pipeline tracks (via scripts/evaluate_pipeline.py)
  * sensitivity: excluding test images that duplicate a training image (label-noise finding)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from vehicle_recognition.calibration.calibration import calibration_report  # noqa: E402
from vehicle_recognition.calibration.selective import selective_summary  # noqa: E402
from vehicle_recognition.config import get_data_paths  # noqa: E402
from vehicle_recognition.evaluation.metrics import classification_report  # noqa: E402
from vehicle_recognition.inference.manifest import load_manifest, sha256_file  # noqa: E402

RES = ROOT / "reports" / "results"


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun-disclosed", default=None, help="reason; required to run again (recorded in the result)")
    a = ap.parse_args()
    if (RES / "final_test.json").exists() and not a.rerun_disclosed:
        raise SystemExit("final_test.json exists: the frozen test evaluation has already been run once")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("Working tree has uncommitted changes: commit (freeze) before the final test evaluation")
    dp = get_data_paths()
    bundle = ROOT / "artifacts" / "deployment"
    m, _ = load_manifest(bundle / "manifest.json")
    decision = json.loads((ROOT / "artifacts" / "decision.json").read_text())
    frozen = {"frozen_commit": git("rev-parse", "HEAD"), "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "manifest_sha256": sha256_file(bundle / "manifest.json"), "model_bundle_id": m.model_bundle_id,
              "decision": decision,
              "config_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT / "configs").glob("*"))},
              "classical_model_sha256": sha256_file(ROOT / "artifacts" / "classical" / "classical_svm.joblib"),
              "rerun_disclosure": a.rerun_disclosed}
    (RES / "freeze_record.json").write_text(json.dumps(frozen, indent=1))
    print("[freeze] recorded", frozen["frozen_commit"])

    # ---- 1) pipeline tracks on test (detector + classifier from the bundle) ----
    t = m.decision.temperature
    cmd = [sys.executable, str(ROOT / "scripts" / "evaluate_pipeline.py"), "--split", "test", "--final-test",
           "--detector", str(bundle / "detector.onnx"), "--det-conf", str(m.detector.confidence_threshold),
           "--det-iou", str(m.detector.nms_iou_threshold), "--classifier", str(bundle / "classifier.onnx"),
           "--img-size", str(m.classifier.input_size), "--temperature", str(t), "--roi-pad-frac",
           str(m.classifier.roi_pad_frac), "--name", "final", "--save-logits"]
    if subprocess.run(cmd).returncode != 0:
        raise SystemExit("pipeline evaluation failed")
    pipe = json.loads((RES / "pipeline" / "test_final.json").read_text())
    z = np.load(dp.processed / "pipeline_test_final.npz", allow_pickle=False)
    y, paths = z["labels"], z["relative_path"]
    p_dl = softmax(z["logits_gt"] / t, axis=1)

    # ---- 2) classical branch on test GT ROIs ----
    from run_classical import extract_split

    from vehicle_recognition.classical.features import FeatureConfig
    from vehicle_recognition.classical.model import load_model, predict_proba_canonical, stack_families

    cb = load_model(ROOT / "artifacts" / "classical" / "classical_svm.joblib")
    canon = pd.read_csv(dp.metadata / "canonical_annotations.csv").set_index("relative_path").loc[paths].reset_index()
    feats = extract_split(canon, dp.raw_extracted, FeatureConfig.from_dict(cb["feature_config"]), 0.05, 4)
    X, _ = stack_families(feats, tuple(cb["spec"]["families"]))
    p_cv_raw = predict_proba_canonical(cb["pipeline"], X)
    ct = decision["classical_temperature"]
    p_cv = softmax(np.log(np.clip(p_cv_raw, 1e-12, 1)) / ct, axis=1)
    alpha = decision.get("fusion_alpha")
    a_eval = alpha if alpha is not None else json.loads((RES / "fusion.json").read_text())["alpha"]
    p_h = a_eval * p_dl + (1 - a_eval) * p_cv
    p_h /= p_h.sum(1, keepdims=True)
    p_prod = p_h if m.decision.mode == "hybrid" else p_dl

    def block(p, mask=None):
        mask = np.ones(len(y), bool) if mask is None else mask
        return classification_report(p[mask], y[mask]) | calibration_report(p[mask], y[mask])

    dup = pd.read_csv(dp.reports / "test_images_duplicating_train.csv")
    clean = ~np.isin(paths, dup.relative_path.values)
    rows = [
        {"method": f"deep only ({m.classifier.architecture} {m.classifier.input_size}px, T={t:.3f})", "roi": "ground truth"} | block(p_dl),
        {"method": "classical HOG+LBP+HSV -> SVM", "roi": "ground truth"} | block(p_cv),
        {"method": f"hybrid alpha={a_eval}", "roi": "ground truth"} | block(p_h),
        {"method": "deep only, excluding test images that duplicate train", "roi": "ground truth"} | block(p_dl, clean),
    ]
    sel = selective_summary(p_prod, y, m.decision.reject_threshold)
    out = {"split": "OFFICIAL TEST (8,041 images; evaluated once after freeze)", "frozen_commit": frozen["frozen_commit"],
           "model_bundle_id": m.model_bundle_id, "production_mode": m.decision.mode, "rows": rows,
           "selective_frozen_threshold": sel, "pipeline_tracks": pipe.get("tracks"), "detector": pipe["detector"],
           "sensitivity_excluded_images": int((~clean).sum()), "rerun_disclosure": a.rerun_disclosed}
    (RES / "final_test.json").write_text(json.dumps(out, indent=1, default=float))
    np.savez_compressed(dp.processed / "final_test_probs.npz", p_dl=p_dl, p_cv=p_cv, labels=y, relative_path=paths)
    print(pd.DataFrame(rows)[["method", "n", "top1", "top5", "macro_f1", "ece_15"]].to_string(index=False))
    print(json.dumps(sel, indent=1))


if __name__ == "__main__":
    main()
