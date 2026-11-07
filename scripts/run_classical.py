"""Classical CV experiments: feature extraction, family ablation, PCA comparison, preprocessing check,
C selection, final calibrated model + validation probabilities for fusion.

Never touches the official test split (test features are extracted only by scripts/final_evaluation.py).
Outputs: reports/results/classical_ablation.json, DATA_ROOT/processed/features/*.npz,
         DATA_ROOT/processed/classical_val_probs.npz, artifacts/classical/classical_svm.joblib
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vehicle_recognition.calibration.calibration import calibration_report  # noqa: E402
from vehicle_recognition.classical.features import (  # noqa: E402
    FeatureConfig,
    hog_features,
    hsv_features,
    lbp_features,
    prepare_roi,
    to_gray,
)
from vehicle_recognition.classical.model import (  # noqa: E402
    ClassicalSpec,
    build_pipeline,
    predict_proba_canonical,
    save_model,
    stack_families,
)
from vehicle_recognition.config import PROJECT_ROOT, artifact_root, get_data_paths, load_yaml  # noqa: E402
from vehicle_recognition.evaluation.metrics import classification_report  # noqa: E402
from vehicle_recognition.preprocessing.image_io import load_dataset_image  # noqa: E402
from vehicle_recognition.preprocessing.roi import crop_roi  # noqa: E402

RESULTS = PROJECT_ROOT / "reports" / "results"


def extract_features_timed(roi, cfg):
    t0 = time.perf_counter()
    img = prepare_roi(roi, cfg)
    gray = to_gray(img, cfg)
    t1 = time.perf_counter()
    h = np.asarray(hog_features(gray, cfg), dtype=np.float32)
    t2 = time.perf_counter()
    lb = lbp_features(gray, cfg)
    t3 = time.perf_counter()
    hs = hsv_features(img, cfg)
    t4 = time.perf_counter()
    return {"hog": h, "lbp": lb, "hsv": hs}, {"prep": (t1 - t0) * 1e3, "hog": (t2 - t1) * 1e3,
                                             "lbp": (t3 - t2) * 1e3, "hsv": (t4 - t3) * 1e3}


def extract_split(rows: pd.DataFrame, image_root: Path, cfg: FeatureConfig, pad: float, workers: int) -> dict:
    def one(r):
        t = time.perf_counter()
        roi = crop_roi(load_dataset_image(image_root / r.relative_path), (r.x_min, r.y_min, r.x_max, r.y_max), pad)
        t_load = time.perf_counter()
        f, times = extract_features_timed(roi, cfg)
        return f, times, (t_load - t) * 1000

    with ThreadPoolExecutor(workers) as ex:
        out = list(ex.map(one, rows.itertuples(index=False), chunksize=16))
    return {
        "hog": np.stack([o[0]["hog"] for o in out]), "lbp": np.stack([o[0]["lbp"] for o in out]),
        "hsv": np.stack([o[0]["hsv"] for o in out]), "y": rows["class_id"].to_numpy(),
        "relative_path": np.array(rows["relative_path"].tolist(), dtype=str),
        **{f"feat_ms_{k}": np.array([o[1][k] for o in out]) for k in ("prep", "hog", "lbp", "hsv")},
        "feat_ms": np.array([sum(o[1].values()) for o in out]), "load_crop_ms": np.array([o[2] for o in out]),
    }


def get_features(dp, canon, subset, cfg, pad, workers, tag):
    path = dp.processed / "features" / f"{subset}_{tag}.npz"
    if path.exists():
        d = dict(np.load(path, allow_pickle=False))
        return d
    rows = canon[canon["subset"] == subset]
    t = time.time()
    d = extract_split(rows, dp.raw_extracted, cfg, pad, workers)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **d)
    print(f"  features {subset}/{tag}: {len(rows)} in {time.time() - t:.0f}s", flush=True)
    return d


def fit_eval(spec: ClassicalSpec, tr: dict, va: dict) -> tuple[dict, object, np.ndarray]:
    Xtr, dims = stack_families(tr, spec.families)
    Xva, _ = stack_families(va, spec.families)
    t = time.time()
    pipe = build_pipeline(spec, dims).fit(Xtr, tr["y"])
    fit_s = time.time() - t
    t = time.perf_counter()
    p = predict_proba_canonical(pipe, Xva)
    svm_ms = (time.perf_counter() - t) * 1000 / len(Xva)
    rep = classification_report(p, va["y"]) | calibration_report(p, va["y"])
    n_comp = pipe.named_steps["pca"].n_components_ if "pca" in pipe.named_steps else Xtr.shape[1]
    import pickle

    size = len(pickle.dumps(pipe))
    row = {"families": "+".join(spec.families), "pca": spec.pca_components, "dims_in": int(Xtr.shape[1]),
           "dims_svm": int(n_comp), "C": spec.C, **rep, "fit_s": round(fit_s, 1), "svm_ms": round(svm_ms, 3),
           "feat_ms": round(float(np.median(tr["feat_ms_prep"]) + sum(np.median(tr[f"feat_ms_{f}"]) for f in spec.families)), 2),
           "model_bytes": size}
    return row, pipe, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--quick", action="store_true", help="tiny subset smoke test")
    a = ap.parse_args()
    c = load_yaml("classical.yaml")
    dp = get_data_paths(a.data_root)
    canon = pd.read_csv(dp.metadata / "canonical_annotations.csv")
    canon = canon[canon["subset"].isin(["train", "val"])]
    if a.quick:
        canon = canon.groupby(["subset", "class_id"]).head(6)
    base_cfg = FeatureConfig.from_dict(c["feature_config"])
    pad = c["roi_pad_frac"]
    tag = f"v{base_cfg.version}" + ("_quick" if a.quick else "")
    tr = get_features(dp, canon, "train", base_cfg, pad, c["workers"], tag)
    va = get_features(dp, canon, "val", base_cfg, pad, c["workers"], tag)
    feat_ms_total = float(np.median(tr["feat_ms"]))
    print(f"train {len(tr['y'])} val {len(va['y'])}; median feature extraction {feat_ms_total:.1f} ms/ROI", flush=True)
    svm = c["svm"]
    out = {"split": "val (from official train)", "note": "fit on train subset; calibrated RBF-SVM (Platt, 5-fold on train)",
           "feature_config": base_cfg.to_dict(), "median_feature_ms_per_roi_all_families": feat_ms_total, "rows": []}

    def spec(fams, pca, C):
        return ClassicalSpec(families=tuple(fams), pca_components=pca, C=C, gamma=svm["gamma"],
                             calibration=svm["calibration"], calibration_cv=svm["calibration_cv"])

    # 1) C selection on the full feature set (no PCA)
    best_c, best_row = None, None
    for C in svm["C_grid"]:
        row, _, _ = fit_eval(spec(["hog", "lbp", "hsv"], None, C), tr, va)
        row["experiment"] = "C_selection"
        out["rows"].append(row)
        print(json.dumps({k: row[k] for k in ("families", "C", "top1", "top5", "macro_f1", "ece_15", "fit_s")}), flush=True)
        if best_row is None or row["top1"] > best_row["top1"]:
            best_c, best_row = C, row
    out["selected_C"] = best_c
    # 2) family ablation (no PCA) at selected C
    for fams in c["ablation"]["families"]:
        if fams == ["hog", "lbp", "hsv"]:
            continue
        row, _, _ = fit_eval(spec(fams, None, best_c), tr, va)
        row["experiment"] = "family_ablation"
        out["rows"].append(row)
        print(json.dumps({k: row[k] for k in ("families", "top1", "top5", "macro_f1", "ece_15", "fit_s")}), flush=True)
    # 3) PCA comparison on the full set
    pca_rows = [best_row]
    for pca in c["ablation"]["pca"]:
        if pca is None:
            continue
        row, _, _ = fit_eval(spec(["hog", "lbp", "hsv"], pca, best_c), tr, va)
        row["experiment"] = "pca_comparison"
        out["rows"].append(row)
        pca_rows.append(row)
        print(json.dumps({k: row[k] for k in ("pca", "dims_svm", "top1", "top5", "ece_15", "fit_s")}), flush=True)
    # PCA explained-variance curve (fit on scaled TRAIN features only)
    from sklearn.decomposition import PCA

    from vehicle_recognition.classical.model import FamilyScaler

    Xtr, dims = stack_families(tr, ("hog", "lbp", "hsv"))
    pc = PCA(svd_solver="full").fit(FamilyScaler(dims).fit_transform(Xtr))
    out["pca_curve"] = np.round(np.cumsum(pc.explained_variance_ratio_)[:1024], 5).tolist()
    # 4) controlled preprocessing check (CLAHE) on the full set, no PCA, selected C
    for pp in c["preprocessing_check"]:
        if not pp["clahe"]:
            continue
        cfg2 = replace(base_cfg, clahe=True)
        tr2 = get_features(dp, canon, "train", cfg2, pad, c["workers"], tag + "_clahe")
        va2 = get_features(dp, canon, "val", cfg2, pad, c["workers"], tag + "_clahe")
        row, _, _ = fit_eval(spec(["hog", "lbp", "hsv"], None, best_c), tr2, va2)
        row["experiment"] = "preprocessing_clahe"
        out["rows"].append(row)
        print(json.dumps({k: row[k] for k in ("experiment", "top1", "top5", "ece_15")}), flush=True)
    # 5) final selection: best validation top-1 among full-set variants (PCA vs none); ties -> smaller model
    final = max(pca_rows, key=lambda r: (round(r["top1"], 4), -r["model_bytes"]))
    out["selected"] = {"families": final["families"], "pca": final["pca"], "C": best_c, "clahe": False,
                       "rule": "max validation top-1 over full-feature PCA variants; ties -> smaller model"}
    fin_spec = spec(["hog", "lbp", "hsv"], final["pca"], best_c)
    row, pipe, p_val = fit_eval(fin_spec, tr, va)
    from vehicle_recognition.data.annotations import build_class_mapping, load_class_names
    from vehicle_recognition.evaluation.metrics import confusion, per_class_accuracy, top_confusions

    names = [ci.name for ci in build_class_mapping(load_class_names(dp.devkit / "cars_meta.mat"))]
    pca_acc = per_class_accuracy(p_val.argmax(1), va["y"], 196)
    out["selected_metrics"] = {k: row[k] for k in ("top1", "top5", "macro_f1", "mean_per_class_acc", "worst_class_acc",
                                                  "ece_15", "nll", "brier", "svm_ms", "feat_ms", "model_bytes")}
    out["selected_per_class_acc"] = {names[i]: (None if np.isnan(v) else round(float(v), 4)) for i, v in enumerate(pca_acc)}
    out["selected_top_confusions"] = top_confusions(confusion(p_val, va["y"]), names, 15)
    if not a.quick:
        art = artifact_root() / "classical"
        art.mkdir(parents=True, exist_ok=True)
        save_model(art / "classical_svm.joblib", pipe, fin_spec, base_cfg.to_dict(), list(stack_families(tr, fin_spec.families)[1]),
                   {"val_top1": row["top1"], "roi_pad_frac": pad})
        np.savez_compressed(dp.processed / "classical_val_probs.npz", probs=p_val, labels=va["y"],
                            relative_path=va["relative_path"])
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / "classical_ablation.json").write_text(json.dumps(out, indent=1, default=float))
    print("selected", out["selected"], "val top1", row["top1"])


if __name__ == "__main__":
    main()
