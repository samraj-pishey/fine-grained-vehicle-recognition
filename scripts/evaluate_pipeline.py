"""Detector + classifier evaluation on ORIGINAL images, three tracks (spec 03 §5, 07 §2).

  Track A  classifier on ground-truth ROI                     (denominator: all images)
  Track B  classifier on the detector box matched to GT        (match: IoU >= 0.5, highest-scoring match;
           denominator reported both as matched-only and as all images with misses counted as errors)
  Track C  end-to-end: system default selection (highest-confidence box) -> classify; success requires
           IoU(selected, GT) >= 0.5 AND correct top-1 class (denominator: all images)

Detector diagnostics at the production confidence threshold: detections per image, best IoU distribution,
miss rate (no detection with IoU >= 0.5), unmatched detections (NOTE: Stanford Cars annotates one car per image;
unmatched boxes may be genuine but unannotated background cars, so they are reported, not called errors).

--split val is used for model selection; --split test is refused unless --final-test is given (frozen run).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vehicle_recognition.config import PROJECT_ROOT, get_data_paths  # noqa: E402
from vehicle_recognition.detection.onnx_detector import OnnxDetector  # noqa: E402
from vehicle_recognition.evaluation.metrics import classification_report  # noqa: E402
from vehicle_recognition.inference.pipeline import OnnxClassifier  # noqa: E402
from vehicle_recognition.preprocessing.image_io import load_dataset_image  # noqa: E402
from vehicle_recognition.preprocessing.roi import crop_roi, iou  # noqa: E402

MATCH_IOU = 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--final-test", action="store_true", help="required to touch the official test split")
    ap.add_argument("--detector", required=True)
    ap.add_argument("--det-conf", type=float, default=0.25)
    ap.add_argument("--det-iou", type=float, default=0.7)
    ap.add_argument("--classifier", default=None, help="ONNX classifier; omit for detector-only evaluation")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--roi-pad-frac", type=float, default=0.05)
    ap.add_argument("--name", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-logits", action="store_true")
    a = ap.parse_args()
    if a.split == "test" and not a.final_test:
        raise SystemExit("Refusing to evaluate on the official test split without --final-test (frozen protocol)")
    dp = get_data_paths()
    canon = pd.read_csv(dp.metadata / "canonical_annotations.csv")
    rows = canon[canon["subset"] == a.split]
    if a.limit:
        rows = rows.head(a.limit)
    det = OnnxDetector(a.detector, conf_thr=a.det_conf, iou_thr=a.det_iou)
    clf = OnnxClassifier(a.classifier, a.img_size) if a.classifier else None
    recs, L_gt, L_pred, L_e2e = [], [], [], []
    t_det, t_cls = [], []
    for r in rows.itertuples(index=False):
        img = load_dataset_image(dp.raw_extracted / r.relative_path)
        h, w = img.shape[:2]
        gt = (max(0.0, r.x_min), max(0.0, r.y_min), min(float(w), r.x_max), min(float(h), r.y_max))
        t = time.perf_counter()
        dets = det.detect(img)
        t_det.append((time.perf_counter() - t) * 1000)
        ious = [iou(d.box, gt) for d in dets]
        best_iou = max(ious) if ious else 0.0
        matched = [d for d, v in zip(dets, ious, strict=True) if v >= MATCH_IOU]
        match = max(matched, key=lambda d: d.score) if matched else None
        top = dets[0] if dets else None
        rec = {"relative_path": r.relative_path, "class_id": int(r.class_id), "n_det": len(dets), "best_iou": best_iou,
               "matched": match is not None, "top_iou": ious[0] if ious else 0.0,
               "top_score": top.score if top else None, "n_unmatched": len(dets) - len(matched),
               "gt_area_frac": (gt[2] - gt[0]) * (gt[3] - gt[1]) / (w * h)}
        if clf is not None:
            t = time.perf_counter()
            lg = clf.logits(crop_roi(img, gt, a.roi_pad_frac))
            t_cls.append((time.perf_counter() - t) * 1000)
            L_gt.append(lg)
            rec["pred_gt"] = int(lg.argmax())
            lp = clf.logits(crop_roi(img, match.box, a.roi_pad_frac)) if match else None
            L_pred.append(lp if lp is not None else np.full(196, np.nan))
            rec["pred_matched"] = int(lp.argmax()) if lp is not None else None
            if top is not None:
                le = lp if (match is not None and top is match) else clf.logits(crop_roi(img, top.box, a.roi_pad_frac))
                rec["pred_e2e"] = int(le.argmax())
                rec["conf_e2e"] = float(softmax(le / a.temperature).max())
                L_e2e.append(le)
            else:
                rec["pred_e2e"] = None
                L_e2e.append(np.full(196, np.nan))
        recs.append(rec)
    df = pd.DataFrame(recs)
    n = len(df)
    detector = {
        "images": n, "conf_threshold": a.det_conf, "match_iou": MATCH_IOU,
        "miss_rate": float((~df.matched).mean()), "no_detection_rate": float((df.n_det == 0).mean()),
        "iou_mean": float(df.best_iou.mean()), "iou_median": float(df.best_iou.median()),
        "iou_p10": float(df.best_iou.quantile(0.10)), "top_box_iou_ge_0.5": float((df.top_iou >= MATCH_IOU).mean()),
        "dets_per_image": df.n_det.value_counts().sort_index().to_dict(),
        "images_with_unmatched_dets": float((df.n_unmatched > 0).mean()),
        "cpu_det_ms_p50": float(np.percentile(t_det, 50)), "cpu_det_ms_p95": float(np.percentile(t_det, 95)),
        "iou_hist": {"bins": np.round(np.linspace(0.025, 0.975, 20), 3).tolist(),
                     "counts": np.histogram(df.best_iou, bins=20, range=(0, 1))[0].tolist()},
    }
    out = {"split": a.split if a.split == "val" else "OFFICIAL TEST", "name": a.name, "detector_path": Path(a.detector).name,
           "detector": detector}
    if clf is not None:
        y = df.class_id.to_numpy()
        Lg = np.stack(L_gt)
        pg = softmax(Lg / a.temperature, axis=1)
        trackA = classification_report(pg, y)
        m = df.matched.to_numpy()
        Lp = np.stack(L_pred)
        trackB_matched = classification_report(softmax(Lp[m] / a.temperature, axis=1), y[m]) if m.any() else None
        trackB_all_top1 = float(((df.pred_matched == df.class_id) & m).mean())
        changed = float((df.pred_matched[m] != df.pred_gt[m]).mean()) if m.any() else None
        e2e_ok = (df.top_iou >= MATCH_IOU) & (df.pred_e2e == df.class_id)
        out["tracks"] = {
            "A_gt_roi": trackA | {"denominator": n},
            "B_pred_roi_matched_only": (trackB_matched or {}) | {"denominator": int(m.sum())},
            "B_pred_roi_all_images_top1": {"top1": trackB_all_top1, "denominator": n, "misses_counted_as_errors": True},
            "B_crop_induced_prediction_changes": {"rate": changed, "denominator": int(m.sum())},
            "C_end_to_end": {"success_rate": float(e2e_ok.mean()), "denominator": n,
                             "class_acc_given_correct_box": float((df.pred_e2e == df.class_id)[df.top_iou >= MATCH_IOU].mean()),
                             "rule": "top-confidence box with IoU>=0.5 to GT AND correct top-1"},
        }
        out["cpu_cls_ms_p50"] = float(np.percentile(t_cls, 50))
        if a.save_logits:
            np.savez_compressed(dp.processed / f"pipeline_{a.split}_{a.name}.npz", logits_gt=Lg, logits_pred=Lp,
                                logits_e2e=np.stack(L_e2e), labels=y, relative_path=np.array(df.relative_path.tolist(), dtype=str))
    dp.processed.mkdir(parents=True, exist_ok=True)
    df.to_csv(dp.processed / f"pipeline_{a.split}_{a.name}_records.csv", index=False)
    res_dir = PROJECT_ROOT / "reports" / "results" / "pipeline"
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / f"{a.split}_{a.name}.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != "detector"} | {"detector": {k: v for k, v in detector.items() if k != "iou_hist"}},
                     indent=1, default=float))


if __name__ == "__main__":
    main()
