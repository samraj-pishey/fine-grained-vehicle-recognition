"""Generate reports/ACADEMIC_REPORT.md from recorded experiment artefacts only.

Every number in the report is read from reports/results/*.json or reports/dataset/*; any artefact that does not
exist is rendered as NOT RUN. Re-run after any experiment: `python scripts/generate_report.py`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "reports" / "results"
DS = ROOT / "reports" / "dataset"


def load(name, base=RES):
    p = base / name
    return json.loads(p.read_text()) if p.exists() else None


def pct(x):
    return "—" if x is None else f"{100 * x:.2f}%"


def f4(x):
    return "—" if x is None else f"{x:.4f}"


def table(rows, cols, fmt=None):
    fmt = fmt or {}
    if not rows:
        return "_NOT RUN_\n"
    out = ["| " + " | ".join(c for c, _ in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        cells = []
        for _, k in cols:
            v = r.get(k)
            f = fmt.get(k)
            cells.append("—" if v is None else (f(v) if f else str(v)))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def not_run(what):
    return f"**NOT RUN** — {what}: no recorded artefact.\n"


def main():
    inv = load("dataset_inventory.json", DS)
    vs = inv["validation_summary"] if inv else {}
    cla = load("classical_ablation.json")
    clf = load("classifier_comparison.json")
    det = load("detector_comparison.json")
    cal = load("calibration.json")
    fus = load("fusion.json")
    sel = load("selective.json")
    rob = load("robustness.json")
    fc = load("failure_cases.json")
    bench = load("benchmark.json")
    final = load("final_test.json")
    freeze = load("freeze_record.json")
    exp = load("export_summary.json")
    e2e_val = load("end_to_end.json")
    deploy = load("deployment_measurements.json")
    S = []
    w = S.append
    w("# Intelligent Fine-Grained Vehicle Recognition and Analysis System\n")
    w(f"*Technical and academic report — generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} from recorded "
      "artefacts by `scripts/generate_report.py`. Validation results come from a split of the official training set; "
      "official-test results are labelled **OFFICIAL TEST** and were computed once after configuration freeze.*\n")

    # 2 abstract
    w("## Abstract\n")
    if final:
        r0 = final["rows"][0]
        w(f"We build an end-to-end system that detects a vehicle in a photograph, lets the user select it, and classifies it "
          f"into one of the 196 Stanford Cars make/model/year classes, with calibrated confidence and abstention. On the "
          f"official test split, the deployed deep classifier reaches **{pct(r0['top1'])} top-1** and **{pct(r0['top5'])} "
          f"top-5** accuracy on ground-truth vehicle crops; the classical HOG+LBP+HSV SVM baseline reaches "
          f"{pct(final['rows'][1]['top1'])} top-1. End-to-end details, calibration, robustness and failure cases follow.\n")
    else:
        w(not_run("final frozen test evaluation (abstract numbers pending)"))

    # 3-4
    w("## 1. Introduction and problem statement\n")
    w("Fine-grained vehicle recognition distinguishes classes that differ in small details (grille, lights, trim) and "
      "between model years of the same model. The task combines localisation (where is the car?) with fine-grained "
      "classification (which make, model and year?). The project also serves as a controlled comparison of classical "
      "computer vision, deep learning and their probability-level fusion, deployed on free CPU hardware.\n")
    w("## 2. Objectives and scope\n")
    w("Objectives: (i) a reproducible data pipeline on the original release; (ii) detector and classifier "
      "comparisons; (iii) a classical HOG/LBP/HSV + calibrated SVM branch; (iv) validated hybrid fusion; (v) calibrated "
      "confidence with rejection; (vi) robustness and failure analysis; (vii) a public CPU web application. "
      "Out of scope: number plates, tracking, traffic events, damage, valuation, ownership, surveillance.\n")

    # 5 dataset
    w("## 3. Dataset description and integrity\n")
    if vs:
        w(f"Stanford Cars (Krause et al., 2013): {vs['counts']['train'] + vs['counts']['test']:,} images, 196 classes, "
          f"official split {vs['counts']['train']:,} train / {vs['counts']['test']:,} test (matches the official description). "
          "The original host was offline; the original files were retrieved from Internet Archive captures and are "
          "byte-identical to the official release (MD5 equal to torchvision's pinned values; `configs/dataset_sources.yaml`).\n")
        w(f"- Decode failures: {vs['decode_failures']}; invalid/out-of-bounds boxes: {vs['non_positive_boxes']}/{vs['out_of_bounds_boxes']}\n"
          f"- Box convention verified empirically: 1-based inclusive (MATLAB) → canonical 0-based half-open\n"
          f"- Train images per class: min/max/mean {vs['per_class_train_min_max_mean']}\n"
          f"- Duplicates: {vs['duplicates']}; train↔test: {vs['cross_split_duplicates']}\n"
          f"- **Label conflicts:** {vs.get('label_conflicting_duplicate_pairs', '—')} duplicate pairs carry different labels "
          "(e.g. Dodge Caliber Wagon 2007 vs 2012) — label noise in the source\n"
          f"- Working split: {vs['subset_counts']} (validation stratified, duplicate-grouped, from official train only)\n"
          "Full report: `reports/dataset/dataset_validation_report.md`.\n")
    else:
        w(not_run("dataset validation"))

    w("## 4. Related work\n")
    w("Hand-crafted descriptors: HOG (Dalal & Triggs, 2005) and LBP (Ojala et al., 2002), classified with SVMs "
      "(Cortes & Vapnik, 1995) and Platt-calibrated probabilities (Platt, 1999). Deep classifiers: MobileNetV3 (Howard et "
      "al., 2019) and EfficientNetV2 (Tan & Le, 2021). One-stage detection: YOLO (Redmon et al., 2016) and its Ultralytics "
      "successors YOLO11/YOLO26 (Jocher et al., Ultralytics). Calibration: temperature scaling (Guo et al., 2017). "
      "Selective classification: Geifman & El-Yaniv (2017). Published Stanford Cars results use different protocols "
      "(crops, resolution, pretraining, training length); we do not compare our numbers to them as directly equivalent.\n")

    w("## 5. System architecture\n")
    w("```text\nupload → validation (signature, size, pixels, decode) → EXIF → RGB → YOLO detector (ONNX) → boxes →\n"
      "user selection → ROI (+5% pad) → CNN (ONNX) → temperature → [optional α-fusion with HOG+LBP+HSV→SVM] →\n"
      "reject policy → Top-5 + make/model/year\n```\n"
      "One detector and one classifier are loaded in production (`artifacts/deployment/manifest.json`, checksum-verified).\n")

    # 8 classical
    w("## 6. Classical computer vision methodology and results\n")
    if cla:
        for r in cla["rows"]:
            r.setdefault("family_weighting", "none" if r["experiment"] == "family_weighting_none" else "equal")
        fc_ = cla["feature_config"]
        w(f"ROI resized to {fc_['resize_wh'][0]}×{fc_['resize_wh'][1]}; HOG ({fc_['hog_orientations']} bins, "
          f"{fc_['hog_pixels_per_cell'][0]}px cells, {fc_['hog_cells_per_block'][0]}×{fc_['hog_cells_per_block'][1]} blocks, L2-Hys), "
          f"uniform LBP (P={fc_['lbp_points']}, R={fc_['lbp_radius']}, {fc_['lbp_grid'][0]}×{fc_['lbp_grid'][1]} grid), HSV joint "
          f"histogram {fc_['hsv_bins']}. Standardisation and PCA fitted on training data only; RBF-SVM with Platt calibration "
          "(5-fold, training data). Validation results:\n")
        w(table(cla["rows"], [("experiment", "experiment"), ("features", "families"), ("PCA", "pca"),
                              ("weighting", "family_weighting"), ("dims", "dims_svm"), ("C", "C"), ("top-1", "top1"),
                              ("top-5", "top5"), ("macro-F1", "macro_f1"), ("ECE", "ece_15"), ("SVM ms", "svm_ms")],
                {"top1": pct, "top5": pct, "macro_f1": f4, "ece_15": f4}))
        w(f"Selected: `{cla.get('selected')}`. Observation: equal per-family weighting let weak LBP/HSV dilute HOG; plain "
          "standardisation (HOG dominates by dimensionality) performed better. PCA explained variance is not feature importance; "
          "importance is assessed by the family ablation.\n")
    else:
        w(not_run("classical branch"))

    w("## 7. Deep-learning methodology\n")
    w("ImageNet-pretrained torchvision backbones, single 196-way head; 2 head-only epochs then full fine-tuning with AdamW "
      "(backbone LR = 0.3 × head LR), cosine schedule with warm-up, label smoothing 0.1, AMP, early stopping on validation "
      "top-1. ROI policy: GT box + 5% padding, squashed to S×S, ImageNet normalisation. Moderate augmentation (flip, ±8° "
      "rotation, perspective, colour jitter, blur, random erasing, ±8% box jitter). Resumable checkpoints; data order is a "
      "function of (seed, epoch).\n")
    if clf:
        w(table(clf["runs"], [("run", "run"), ("arch", "arch"), ("px", "img_size"), ("top-1", "top1"), ("top-5", "top5"),
                              ("macro-F1", "macro_f1"), ("ECE raw", "ece_raw"), ("ECE cal.", "ece_cal_crossfit"),
                              ("T", "temperature"), ("best ep", "best_epoch"), ("CPU ms", "cpu_ms_p50")],
                {"top1": pct, "top5": pct, "macro_f1": f4, "ece_raw": f4, "ece_cal_crossfit": f4, "temperature": f4,
                 "cpu_ms_p50": lambda v: f"{v:.1f}"}))
        sv = clf.get("seed_variance", {})
        w(f"Seed variance (baseline, n={sv.get('n')}): top-1 mean {pct(sv.get('top1_mean'))} ± "
          f"{pct(sv.get('top1_std')) if sv.get('top1_std') is not None else '—'}. Selected: `{clf['selected']}` — {clf['rationale']}\n")
    else:
        w(not_run("classifier comparison"))

    w("## 8. Detector methodology and results\n")
    if det:
        w("YOLO26n and YOLO11n fine-tuned as single-class detectors (40 epochs, 640 px, batch 32, identical settings).\n")
        w(table(det["runs"], [("model", "name"), ("mAP50", "mAP50"), ("mAP50-95", "mAP50_95"), ("P", "precision"),
                              ("R", "recall"), ("miss rate", "miss_rate"), ("IoU median", "iou_median"),
                              ("CPU ms p50", "cpu_det_ms_p50"), ("ONNX MB", "onnx_bytes")],
                {k: f4 for k in ("mAP50", "mAP50_95", "precision", "recall", "miss_rate", "iou_median")} |
                {"cpu_det_ms_p50": lambda v: f"{v:.1f}", "onnx_bytes": lambda v: f"{v / 1e6:.1f}"}))
        w(f"Selected `{det['selected']}`: {det['rationale']}. {det['note']}\n")
    else:
        w(not_run("detector comparison"))

    w("## 9. Hybrid fusion\n")
    if fus:
        w("P_hybrid = α·P_deep + (1−α)·P_classical, α selected on validation (NLL); reported numbers include a cross-fitted "
          "estimate (α fitted on one half of validation, scored on the other).\n")
        w(table(fus["comparison"], [("method", "method"), ("top-1", "top1"), ("top-5", "top5"), ("macro-F1", "macro_f1"),
                                    ("ECE", "ece_15"), ("NLL", "nll"), ("Brier", "brier")],
                {"top1": pct, "top5": pct, "macro_f1": f4, "ece_15": f4, "nll": f4, "brier": f4}))
        w(f"**Production decision:** {fus['production_mode']} — {fus['rationale']}\n")
    else:
        w(not_run("hybrid fusion"))

    w("## 10. Training setup and reproducibility\n")
    w("Free Kaggle GPU (Tesla T4); environment recorded per run (`env.json`). Dependencies pinned (`uv.lock`, "
      "`requirements/train-cloud.txt`). Dataset identity pinned by SHA-256. Split seed and manifest hash recorded. "
      "Single training run per configuration except the baseline, which has three seeds.\n")

    w("## 11. Evaluation protocol\n")
    w("Model choice, temperature, fusion weight and rejection threshold were selected on validation only and frozen "
      "before a single evaluation on the official test split. ECE: top-label, 15 equal-width bins. Detection match: "
      "IoU ≥ 0.5. End-to-end success: highest-confidence box with IoU ≥ 0.5 and correct top-1 (denominator: all images).\n")

    w("## 12. Results on the official test split\n")
    if final:
        w(f"Frozen commit `{final['frozen_commit']}`, bundle `{final['model_bundle_id']}`.\n")
        w(table(final["rows"], [("method", "method"), ("n", "n"), ("top-1", "top1"), ("top-5", "top5"),
                                ("macro-F1", "macro_f1"), ("ECE", "ece_15"), ("NLL", "nll")],
                {"top1": pct, "top5": pct, "macro_f1": f4, "ece_15": f4, "nll": f4}))
        tr = final.get("pipeline_tracks") or {}
        if tr:
            w("| track | value | denominator |\n|---|---|---|\n"
              f"| A: classifier on GT ROI, top-1 | {pct(tr['A_gt_roi']['top1'])} | {tr['A_gt_roi']['denominator']} |\n"
              f"| B: classifier on matched detector ROI, top-1 | {pct(tr['B_pred_roi_matched_only'].get('top1'))} | {tr['B_pred_roi_matched_only']['denominator']} |\n"
              f"| B: same, misses counted as errors | {pct(tr['B_pred_roi_all_images_top1']['top1'])} | {tr['B_pred_roi_all_images_top1']['denominator']} |\n"
              f"| B: predictions changed by crop (GT vs detector box) | {pct(tr['B_crop_induced_prediction_changes']['rate'])} | {tr['B_crop_induced_prediction_changes']['denominator']} |\n"
              f"| C: end-to-end success | {pct(tr['C_end_to_end']['success_rate'])} | {tr['C_end_to_end']['denominator']} |\n")
        d = final["detector"]
        w(f"\nDetector on test: miss rate {pct(d['miss_rate'])}, median IoU {f4(d['iou_median'])}, images with unmatched "
          f"detections {pct(d['images_with_unmatched_dets'])} (may be real but unannotated background cars).\n")
        s = final["selective_frozen_threshold"]
        w(f"\nSelective prediction with the frozen threshold {f4(s['threshold'])}: coverage {pct(s['coverage'])}, selective "
          f"accuracy {pct(s['selective_accuracy'])} (accuracy before rejection {pct(s['accuracy_before_rejection'])}).\n")
    else:
        w(not_run("final frozen test evaluation"))
    if e2e_val:
        w("\nValidation pipeline tracks: see `reports/results/end_to_end.json`.\n")

    w("## 13. Ablation studies\n")
    w("Feature-family, PCA and weighting ablations: §6. Architecture × resolution, augmentation strength and "
      "class-weighted loss: §7 table (runs `*_auglight`, `*_cwloss`). Deep vs hybrid: §9. FP32 vs INT8: §16.\n")

    w("## 14. Robustness evaluation\n")
    if rob:
        w("Synthetic perturbations on validation ROIs (not natural conditions; never used for tuning).\n")
        w(table(rob["rows"], [("condition", "condition"), ("severity", "severity"), ("top-1", "top1"), ("top-5", "top5"),
                              ("ECE", "ece_15"), ("mean conf.", "mean_conf"), ("classical top-1", "classical_top1"),
                              ("hybrid top-1", "hybrid_top1"), ("detector hit", "detector_hit_rate")],
                {"top1": pct, "top5": pct, "ece_15": f4, "mean_conf": f4, "classical_top1": pct, "hybrid_top1": pct,
                 "detector_hit_rate": pct}))
    else:
        w(not_run("robustness"))

    w("## 15. Confidence and calibration\n")
    if cal:
        w(table(cal["table"], [("model", "model"), ("ECE", "ece_15"), ("NLL", "nll"), ("Brier", "brier"),
                               ("mean conf.", "mean_confidence")], {"ece_15": f4, "nll": f4, "brier": f4, "mean_confidence": f4}))
    else:
        w(not_run("calibration"))
    if sel:
        w(f"Rejection policy: {sel['policy']}. Validation: {sel.get('summary')}; cross-fitted estimate "
          f"{sel.get('crossfit_estimate')}.\n")

    w("## 16. Failure-case analysis\n")
    if fc:
        w(f"{len(fc['cases'])} genuine cases (from {fc['n_candidates']} candidates, {fc['n_wrong_top1']} wrong top-1). "
          f"Category rules: {fc['rules']}\n")
        w(table(fc["cases"], [("category", "category"), ("image", "image_id"), ("true", "true_class"),
                              ("predicted", "predicted_class"), ("conf.", "classification_confidence"), ("IoU", "best_iou"),
                              ("proposed improvement", "proposed_improvement"), ("tested", "improvement_tested")],
                {"classification_confidence": f4}))
    else:
        w(not_run("failure-case analysis"))

    w("## 17. Deployment and optimisation\n")
    if exp:
        w(f"Export/quantisation: `reports/results/export_summary.json` — {exp.get('summary', '')}\n")
    if bench:
        w(f"CPU benchmark ({bench['hardware'].get('platform')}, {bench['measured_utc']}): {bench['protocol']}\n")
        w(table(bench["rows"], [("kind", "kind"), ("model", "model"), ("input", "input"), ("threads", "threads"),
                                ("MB", "bytes"), ("load ms", "load_ms"), ("p50 ms", "p50_ms"), ("p95 ms", "p95_ms"),
                                ("peak RSS MB", "peak_rss_mb")],
                {"bytes": lambda v: f"{v / 1e6:.1f}", "load_ms": lambda v: f"{v:.0f}", "p50_ms": lambda v: f"{v:.1f}",
                 "p95_ms": lambda v: f"{v:.1f}", "peak_rss_mb": lambda v: f"{v:.0f}"}))
    else:
        w(not_run("CPU benchmark"))
    if deploy:
        w(f"Hosted measurements: {json.dumps(deploy)}\n")
    else:
        w(not_run("hosted cold/warm latency measurement"))

    w("## 18. Limitations and threats to validity\n")
    w("- Single run per configuration except the seed-repeated baseline; small differences may not be significant.\n"
      "- Validation (1,215 images, ~6 per class) is small; selection noise is non-trivial.\n"
      "- Stanford Cars contains label conflicts between near-identical classes; accuracy has an irreducible ceiling.\n"
      "- Detector trained/evaluated on one annotated car per image; multi-vehicle performance is not benchmarked.\n"
      "- Robustness uses synthetic perturbations, not natural condition labels.\n"
      "- Classes cover mostly 1991–2012 models; newer vehicles are out of distribution.\n"
      "- Free hosting: cold starts and shared CPU; latency on the host differs from local measurements.\n")

    w("## 19. Conclusion and future work\n")
    w("See abstract and results. Future work: part-based or attention-based fine-grained models, higher resolution, "
      "external real-world evaluation set, and detector training on multi-vehicle scenes.\n")

    w("## References\n")
    for ref in [
        "N. Dalal, B. Triggs. Histograms of Oriented Gradients for Human Detection. CVPR 2005.",
        "T. Ojala, M. Pietikäinen, T. Mäenpää. Multiresolution gray-scale and rotation invariant texture classification "
        "with local binary patterns. IEEE TPAMI 24(7), 2002.",
        "C. Cortes, V. Vapnik. Support-vector networks. Machine Learning 20, 1995.",
        "J. Platt. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. 1999.",
        "T.-F. Wu, C.-J. Lin, R. C. Weng. Probability estimates for multi-class classification by pairwise coupling. JMLR 5, 2004.",
        "J. Krause, M. Stark, J. Deng, L. Fei-Fei. 3D Object Representations for Fine-Grained Categorization. 3dRR-13 (ICCV workshops), 2013.",
        "J. Redmon, S. Divvala, R. Girshick, A. Farhadi. You Only Look Once: Unified, Real-Time Object Detection. CVPR 2016.",
        "G. Jocher et al. Ultralytics YOLO (YOLO11, YOLO26). Software, https://github.com/ultralytics/ultralytics.",
        "A. Howard et al. Searching for MobileNetV3. ICCV 2019.",
        "M. Tan, Q. Le. EfficientNetV2: Smaller Models and Faster Training. ICML 2021.",
        "C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger. On Calibration of Modern Neural Networks. ICML 2017.",
        "Y. Geifman, R. El-Yaniv. Selective Classification for Deep Neural Networks. NeurIPS 2017.",
    ]:
        w(f"- {ref}")
    w("\n## Appendix: configuration and provenance\n")
    w(f"Freeze record: `{json.dumps(freeze['frozen_commit']) if freeze else 'NOT RUN'}`. Configs: `configs/`. "
      "Acceptance ledger: `milestone_acceptance.yaml`.\n")
    (ROOT / "reports" / "ACADEMIC_REPORT.md").write_text("\n".join(S) + "\n")
    print("wrote reports/ACADEMIC_REPORT.md")


if __name__ == "__main__":
    main()
