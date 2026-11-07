"""Synthetic robustness evaluation: 6 conditions x 3 severities on the VALIDATION subset (default).

Reports per condition/severity: deep top-1/top-5/ECE/mean confidence, classical and hybrid top-1 (if a classical
model is given), and detector hit rate (IoU>=0.5 to GT) for image-level conditions. The perturbed results are
analysis only; they are never used to tune any model, threshold or fusion weight.
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vehicle_recognition.calibration.calibration import ece  # noqa: E402
from vehicle_recognition.config import PROJECT_ROOT, get_data_paths  # noqa: E402
from vehicle_recognition.evaluation.metrics import topk_accuracy  # noqa: E402
from vehicle_recognition.evaluation.perturbations import CONDITIONS, IMAGE_LEVEL, SEVERITIES, perturb  # noqa: E402
from vehicle_recognition.inference.pipeline import OnnxClassifier  # noqa: E402
from vehicle_recognition.preprocessing.image_io import load_dataset_image  # noqa: E402
from vehicle_recognition.preprocessing.roi import crop_roi, iou  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifier", required=True)
    ap.add_argument("--img-size", type=int, required=True)
    ap.add_argument("--temperature", type=float, required=True)
    ap.add_argument("--detector", default=None)
    ap.add_argument("--classical", default=None, help="classical_svm.joblib (optional)")
    ap.add_argument("--classical-temperature", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--n", type=int, default=600, help="validation images (stratified sample); 0 = all")
    ap.add_argument("--roi-pad-frac", type=float, default=0.05)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--final-test", action="store_true")
    a = ap.parse_args()
    if a.split == "test" and not a.final_test:
        raise SystemExit("Refusing official test split without --final-test")
    dp = get_data_paths()
    canon = pd.read_csv(dp.metadata / "canonical_annotations.csv")
    rows = canon[canon.subset == a.split]
    if a.n and a.n < len(rows):
        k = max(1, round(a.n / 196))  # stratified: k images per class, deterministic
        rows = rows.sample(frac=1.0, random_state=0).groupby("class_id").head(k).sort_index()
    clf = OnnxClassifier(a.classifier, a.img_size)
    det = None
    if a.detector:
        from vehicle_recognition.detection.onnx_detector import OnnxDetector

        det = OnnxDetector(a.detector, conf_thr=0.25)
    cl = None
    if a.classical:
        from vehicle_recognition.classical.features import FeatureConfig, extract_features
        from vehicle_recognition.classical.model import load_model, predict_proba_canonical, stack_families

        b = load_model(a.classical)
        cl = (b["pipeline"], FeatureConfig.from_dict(b["feature_config"]), tuple(b["spec"]["families"]))

    def classical_p(roi):
        f = extract_features(roi, cl[1])
        X, _ = stack_families({k: v[None] for k, v in f.items()}, cl[2])
        p = predict_proba_canonical(cl[0], X)[0]
        return softmax(np.log(np.clip(p, 1e-12, 1)) / a.classical_temperature)

    settings = [("clean", 0)] + [(c, s) for c in CONDITIONS for s in SEVERITIES]
    acc = {k: {"L": [], "C": [], "hit": []} for k in settings}
    y = rows.class_id.to_numpy()
    for i, r in enumerate(rows.itertuples(index=False)):
        img = load_dataset_image(dp.raw_extracted / r.relative_path)
        h, w = img.shape[:2]
        gt = (max(0.0, r.x_min), max(0.0, r.y_min), min(float(w), r.x_max), min(float(h), r.y_max))
        for cond, sev in settings:
            seed = zlib.crc32(f"{r.relative_path}|{cond}|{sev}".encode())
            roi = crop_roi(img, gt, a.roi_pad_frac) if cond == "clean" else perturb(img, gt, cond, sev, seed, a.roi_pad_frac)
            acc[(cond, sev)]["L"].append(clf.logits(roi))
            if cl:
                acc[(cond, sev)]["C"].append(classical_p(roi))
            if det is not None and (cond == "clean" or cond in IMAGE_LEVEL):
                from vehicle_recognition.evaluation.perturbations import IMAGE_LEVEL as IL

                im2 = img if cond == "clean" else IL[cond](img, sev, np.random.default_rng(seed))
                ds = det.detect(im2)
                acc[(cond, sev)]["hit"].append(bool(ds) and iou(ds[0].box, gt) >= 0.5)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(rows)}", flush=True)
    out_rows = []
    for (cond, sev), d in acc.items():
        p = softmax(np.array(d["L"]) / a.temperature, axis=1)
        row = {"condition": cond, "severity": sev, "n": len(y), "top1": float((p.argmax(1) == y).mean()),
               "top5": topk_accuracy(p, y, 5), "ece_15": ece(p, y), "mean_conf": float(p.max(1).mean())}
        if cl:
            pc = np.array(d["C"])
            row["classical_top1"] = float((pc.argmax(1) == y).mean())
            if a.alpha is not None:
                ph = a.alpha * p + (1 - a.alpha) * pc
                row["hybrid_top1"] = float((ph.argmax(1) == y).mean())
        if d["hit"]:
            row["detector_hit_rate"] = float(np.mean(d["hit"]))
        out_rows.append(row)
    out = {"split": f"{a.split} (synthetic perturbations)", "note": "GT ROI for classification; image-level perturbations "
           "also re-run the detector", "classifier": Path(a.classifier).name, "rows": out_rows}
    res = PROJECT_ROOT / "reports" / "results"
    res.mkdir(parents=True, exist_ok=True)
    (res / ("robustness.json" if a.split == "val" else "robustness_test.json")).write_text(json.dumps(out, indent=1))
    print(pd.DataFrame(out_rows).to_string(index=False))


if __name__ == "__main__":
    main()
