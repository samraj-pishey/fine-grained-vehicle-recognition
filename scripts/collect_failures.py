"""Collect genuine failure cases from the frozen evaluation records (analysis only; nothing is tuned on them).

Candidates: wrong top-1 (incl. high-confidence wrong), detector miss / poor IoU, deep-classical disagreement.
Categories use transparent, measurable rules (reported with each case) and are confirmed by visual review:
  similar_generation  wrong class with the same make+model text, different year/variant
  detector_miss       no detection with IoU >= 0.5 (or best IoU < 0.5)
  low_resolution      ground-truth box shorter side < 120 px
  lighting            ROI mean luminance < 60 or > 200
  viewpoint           box aspect ratio w/h < 1.15 (near-frontal / rear view)
  occlusion_truncation  GT box touches the image border (vehicle cut off) - truncation proxy for occlusion
Output: reports/results/failure_cases.json (no images; dataset images may not be redistributed).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vehicle_recognition.config import get_data_paths  # noqa: E402
from vehicle_recognition.preprocessing.image_io import load_dataset_image  # noqa: E402
from vehicle_recognition.preprocessing.roi import crop_roi  # noqa: E402

IMPROVE = {
    "similar_generation": "higher input resolution / part-focused crops (grille, lights) or fine-grained attention; "
                          "more examples of the confusable pair",
    "detector_miss": "lower detector threshold with user confirmation, or train with more small/occluded vehicles",
    "low_resolution": "super-resolution or training-time downscale augmentation; show low-resolution warning",
    "lighting": "stronger photometric augmentation; exposure normalisation",
    "viewpoint": "more frontal/rear training views; viewpoint-balanced sampling",
    "occlusion_truncation": "random-erasing/cutout at higher rates; occlusion-aware training",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=2)
    a = ap.parse_args()
    dp = get_data_paths()
    z = np.load(dp.processed / "final_test_probs.npz", allow_pickle=False)
    p_dl, p_cv, y, paths = z["p_dl"], z["p_cv"], z["labels"], z["relative_path"]
    rec = pd.read_csv(dp.processed / "pipeline_test_final_records.csv").set_index("relative_path").loc[paths].reset_index()
    canon = pd.read_csv(dp.metadata / "canonical_annotations.csv").set_index("relative_path").loc[paths].reset_index()
    classes = json.loads((ROOT / "reports" / "dataset" / "class_mapping.json").read_text())["classes"]
    pred = p_dl.argmax(1)
    conf = p_dl.max(1)
    cand = np.where((pred != y) | (~rec.matched.to_numpy()) | (rec.best_iou.to_numpy() < 0.5))[0]
    rows = []
    for i in cand:
        r, c = canon.iloc[i], rec.iloc[i]
        bw, bh = r.x_max - r.x_min, r.y_max - r.y_min
        tc, pc = classes[int(y[i])], classes[int(pred[i])]
        roi = crop_roi(load_dataset_image(dp.raw_extracted / r.relative_path), (r.x_min, r.y_min, r.x_max, r.y_max), 0.05)
        lum = float(roi.mean())
        cats = []
        if (not c.matched) or c.best_iou < 0.5:
            cats.append("detector_miss")
        if pred[i] != y[i] and tc["make"] == pc["make"] and tc["model"].split()[0] == pc["model"].split()[0]:
            cats.append("similar_generation")
        if min(bw, bh) < 120:
            cats.append("low_resolution")
        if lum < 60 or lum > 200:
            cats.append("lighting")
        if bw / max(1, bh) < 1.15:
            cats.append("viewpoint")
        if r.x_min <= 1 or r.y_min <= 1 or r.x_max >= r.width - 1 or r.y_max >= r.height - 1:
            cats.append("occlusion_truncation")
        top5 = np.argsort(-p_dl[i])[:5]
        rows.append({
            "image_id": r.relative_path, "split": "test", "true_class": tc["name"], "predicted_class": pc["name"],
            "top5": [{"class": classes[k]["name"], "p": round(float(p_dl[i, k]), 4)} for k in top5],
            "classification_confidence": round(float(conf[i]), 4), "detection_confidence": c.top_score,
            "gt_box": [round(v, 1) for v in (r.x_min, r.y_min, r.x_max, r.y_max)], "best_iou": round(float(c.best_iou), 3),
            "classical_prediction": classes[int(p_cv[i].argmax())]["name"],
            "deep_classical_disagree": bool(p_cv[i].argmax() != pred[i]),
            "categories_by_rule": cats, "roi_mean_luminance": round(lum, 1), "box_wh": [round(bw), round(bh)],
            "correct": bool(pred[i] == y[i]),
        })
    df = pd.DataFrame(rows)
    picked, used = [], set()
    for cat in IMPROVE:
        sub = df[df.categories_by_rule.apply(lambda cs, c=cat: c in cs) & ~df.image_id.isin(used)]
        sub = sub.sort_values("classification_confidence", ascending=False)  # most confident failures first
        for _, s in sub.head(a.per_category).iterrows():
            d = s.to_dict()
            d["category"] = cat
            d["diagnosis"] = (f"rule-based category '{cat}'; "
                              + ("high-confidence error" if d["classification_confidence"] >= 0.5 and not d["correct"] else
                                 "detector failure" if cat == "detector_miss" else "low-confidence error"))
            d["proposed_improvement"] = IMPROVE[cat]
            d["improvement_tested"] = False
            picked.append(d)
            used.add(d["image_id"])
    counts = {cat: int(df.categories_by_rule.apply(lambda cs, c=cat: c in cs).sum()) for cat in IMPROVE}
    out = {"split": "OFFICIAL TEST (post-freeze analysis only)", "n_candidates": len(df),
           "n_wrong_top1": int((~df.correct).sum()), "category_counts_among_candidates": counts,
           "rules": __doc__.split("Categories")[1].split("Output")[0].strip(), "cases": picked}
    (ROOT / "reports" / "results" / "failure_cases.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: out[k] for k in ("n_candidates", "n_wrong_top1", "category_counts_among_candidates")}, indent=1))
    print(f"{len(picked)} cases selected")


if __name__ == "__main__":
    main()
