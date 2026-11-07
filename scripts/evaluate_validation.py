"""Validation-only analysis of trained classifiers + classical branch: calibration, comparison, fusion, rejection.

Protocol (no test data involved):
  * Every reported calibration/fusion/threshold number is CROSS-FITTED on validation: parameters fitted on one
    half of the validation set and scored on the other half (2 folds, stratified by class, repeated with both
    halves), so the reported numbers are not optimistic.
  * The FROZEN parameters used in production (T, alpha, reject threshold) are then fitted on the full validation set.

Inputs : --runs <dir with classifiers/<run>/val_logits_best.npz>, DATA_ROOT/processed/classical_val_probs.npz
Outputs: reports/results/{classifier_comparison,calibration,fusion,selective}.json, artifacts/decision.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vehicle_recognition.calibration.calibration import (  # noqa: E402
    apply_temperature,
    calibration_report,
    fit_temperature,
    probs_to_logits,
    reliability_bins,
)
from vehicle_recognition.calibration.selective import coverage_risk_curve, select_threshold, selective_summary  # noqa: E402
from vehicle_recognition.config import PROJECT_ROOT, artifact_root, get_data_paths  # noqa: E402
from vehicle_recognition.evaluation.metrics import classification_report  # noqa: E402
from vehicle_recognition.fusion.fusion import fuse, select_alpha  # noqa: E402

RESULTS = PROJECT_ROOT / "reports" / "results"
TARGET_SELECTIVE_ACC = 0.95  # policy: abstain so that validation selective risk <= 5 %


def halves(y: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = np.zeros(len(y), bool)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        a[idx[: len(idx) // 2]] = True
    return a, ~a


def crossfit_temperature(logits, y) -> dict:
    a, b = halves(y)
    out = []
    for fit, ev in ((a, b), (b, a)):
        t = fit_temperature(logits[fit], y[fit])
        out.append(calibration_report(apply_temperature(logits[ev], t), y[ev]) | {"T": t})
    return {k: float(np.mean([o[k] for o in out])) for k in out[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="directory containing classifiers/<run>/")
    ap.add_argument("--select", default=None, help="force a classifier run name (default: rule below)")
    ap.add_argument("--latency", default=None, help="JSON with CPU latency per run (from benchmark script)")
    a = ap.parse_args()
    dp = get_data_paths()
    runs_dir = Path(a.runs) / "classifiers"
    lat = json.loads(Path(a.latency).read_text()) if a.latency else {}
    rows, cache = [], {}
    for d in sorted(runs_dir.iterdir()):
        f = d / "val_logits_best.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=False)
        logits, y, paths = z["logits"].astype(np.float64), z["labels"], z["relative_path"]
        cfg = json.loads((d / "config.json").read_text())
        hist = pd.read_csv(d / "history.csv")
        p_raw = softmax(logits, axis=1)
        T = fit_temperature(logits, y)
        cf = crossfit_temperature(logits, y)
        rep = classification_report(p_raw, y)
        rows.append({"run": d.name, "arch": cfg["arch"], "img_size": cfg["img_size"], "aug": cfg.get("aug"),
                     "seed": cfg.get("seed"), "class_weighted_loss": cfg.get("class_weighted_loss", False),
                     **rep, "ece_raw": calibration_report(p_raw, y)["ece_15"], "ece_cal_crossfit": cf["ece_15"],
                     "nll_cal_crossfit": cf["nll"], "temperature": T, "params": cfg.get("params"),
                     "best_epoch": int(hist.loc[hist.val_top1.idxmax(), "epoch"]), "epochs_run": len(hist),
                     "gpu": json.loads((d / "env.json").read_text()).get("gpu"),
                     **{k: v for k, v in lat.get(d.name, {}).items()}})
        cache[d.name] = (logits, y, paths, T)
    if not rows:
        raise SystemExit("no classifier runs with val_logits_best.npz found")
    df = pd.DataFrame(rows)
    # seed variance of the baseline configuration
    seeds = df[df.run.str.match(r"^cls_mnv3l_224(_s\d+)?$")]
    seed_stats = {"runs": seeds.run.tolist(), "top1_mean": float(seeds.top1.mean()), "top1_std": float(seeds.top1.std(ddof=1))
                  if len(seeds) > 1 else None, "n": len(seeds)}
    # selection rule among the primary arch x resolution grid (moderate aug, seed 42, no class weighting)
    prim = df[df.run.isin(["cls_mnv3l_224", "cls_mnv3l_288", "cls_effv2s_224", "cls_effv2s_288"])]
    if a.select:
        sel = a.select
        rule = f"forced by --select {sel}"
    else:
        best = prim.top1.max()
        sd = seed_stats["top1_std"] or 0.005
        # candidates statistically indistinguishable from the best (within 2 seed-std) -> prefer the fastest on CPU
        near = prim[prim.top1 >= best - 2 * sd]
        key = "cpu_ms_p50" if "cpu_ms_p50" in near and near.cpu_ms_p50.notna().all() else "params"
        sel = near.sort_values([key, "top1"], ascending=[True, False]).iloc[0].run
        rule = (f"highest validation top-1 = {best:.4f}; runs within 2x the baseline seed std ({sd:.4f}) treated as "
                f"equivalent; among those the lowest {key} wins")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "classifier_comparison.json").write_text(json.dumps({
        "split": "val (from official train)", "note": "ECE: 15 equal-width bins; calibrated ECE is cross-fitted on validation halves",
        "runs": df.round(5).to_dict("records"), "seed_variance": seed_stats, "selected": sel, "rationale": rule}, indent=1, default=float))

    # -------- calibration (selected deep model) and classical alignment --------
    logits, y, paths, T = cache[sel]
    p_dl = apply_temperature(logits, T)
    cz = np.load(dp.processed / "classical_val_probs.npz", allow_pickle=False)
    order = pd.Series(np.arange(len(cz["relative_path"])), index=cz["relative_path"])
    idx = order.reindex(paths).to_numpy()
    if np.isnan(idx.astype(float)).any():
        raise SystemExit("classical and deep validation sets are not aligned (relative_path mismatch)")
    p_cv_raw = cz["probs"][idx.astype(int)]
    assert np.array_equal(cz["labels"][idx.astype(int)], y), "label mismatch between branches"
    Tc = fit_temperature(probs_to_logits(p_cv_raw), y)
    p_cv = apply_temperature(probs_to_logits(p_cv_raw), Tc)
    cal = {"split": "val (from official train)", "ece_definition": "top-label, 15 equal-width bins",
           "table": [
               {"model": f"deep {sel} (raw softmax)"} | calibration_report(softmax(logits, axis=1), y),
               {"model": f"deep {sel} (T={T:.3f}, cross-fitted)"} | {k: v for k, v in crossfit_temperature(logits, y).items() if k != "T"},
               {"model": "classical SVM (Platt)"} | calibration_report(p_cv_raw, y),
               {"model": f"classical SVM (+T={Tc:.3f}, cross-fitted)"} | {k: v for k, v in crossfit_temperature(probs_to_logits(p_cv_raw), y).items() if k != "T"},
           ],
           "reliability": {"deep raw": reliability_bins(softmax(logits, axis=1), y), f"deep T={T:.2f}": reliability_bins(p_dl, y),
                           "classical": reliability_bins(p_cv_raw, y)},
           "frozen": {"deep_temperature": T, "classical_temperature": Tc}}

    # -------- fusion (cross-fitted alpha estimate + frozen alpha on full val) --------
    h1, h2 = halves(y, seed=1)
    cf_rows = []
    for fit, ev in ((h1, h2), (h2, h1)):
        al = select_alpha(p_dl[fit], p_cv[fit], y[fit])["alpha"]
        pf = fuse(p_dl[ev], p_cv[ev], al)
        cf_rows.append(classification_report(pf, y[ev]) | calibration_report(pf, y[ev]) | {"alpha": al})
    hyb_cf = {k: float(np.mean([r[k] for r in cf_rows])) for k in cf_rows[0]}
    dl_half = [classification_report(p_dl[ev], y[ev]) | calibration_report(p_dl[ev], y[ev]) for ev in (h2, h1)]
    dl_cf = {k: float(np.mean([r[k] for r in dl_half])) for k in dl_half[0]}
    sel_alpha = select_alpha(p_dl, p_cv, y)
    alpha = sel_alpha["alpha"]
    p_h = fuse(p_dl, p_cv, alpha)
    comp = [{"method": "deep only (calibrated)"} | classification_report(p_dl, y) | calibration_report(p_dl, y),
            {"method": "classical only (calibrated)"} | classification_report(p_cv, y) | calibration_report(p_cv, y),
            {"method": f"hybrid alpha={alpha} (alpha fit on same val; optimistic)"} | classification_report(p_h, y) | calibration_report(p_h, y),
            {"method": "deep only, half-split eval"} | dl_cf,
            {"method": f"hybrid, cross-fitted alpha (mean alpha={hyb_cf['alpha']:.2f})"} | hyb_cf]
    gain_top1 = hyb_cf["top1"] - dl_cf["top1"]
    gain_nll = dl_cf["nll"] - hyb_cf["nll"]
    # class-wise impact
    from vehicle_recognition.evaluation.metrics import per_class_accuracy

    d_pc = per_class_accuracy(p_dl.argmax(1), y, 196)
    h_pc = per_class_accuracy(p_h.argmax(1), y, 196)
    improved, worsened = int(np.nansum(h_pc > d_pc + 1e-9)), int(np.nansum(h_pc < d_pc - 1e-9))
    use_hybrid = gain_top1 >= 0.005 and gain_nll >= 0 and alpha < 1.0
    rationale = (f"cross-fitted hybrid vs deep: top-1 {gain_top1:+.4f}, NLL {-gain_nll:+.4f}; classes improved/worsened "
                 f"{improved}/{worsened}. Rule: adopt hybrid only if cross-fitted top-1 gain >= 0.5 pp without NLL "
                 f"regression (classical branch adds CPU latency).")
    fusion = {"split": "val (from official train)", "alpha": alpha, "criterion": sel_alpha["criterion"], "grid": sel_alpha["grid"],
              "comparison": comp, "classwise": {"improved": improved, "worsened": worsened},
              "production_mode": "hybrid" if use_hybrid else "deep_only", "rationale": rationale}

    # -------- selective prediction on the production-mode probabilities --------
    p_prod = p_h if use_hybrid else p_dl
    cf_sel = []
    for fit, ev in ((h1, h2), (h2, h1)):
        th = select_threshold(p_prod[fit], y[fit], target_selective_accuracy=TARGET_SELECTIVE_ACC)
        if th["threshold"] is not None:
            cf_sel.append(selective_summary(p_prod[ev], y[ev], th["threshold"]))
    final_th = select_threshold(p_prod, y, target_selective_accuracy=TARGET_SELECTIVE_ACC)
    selective = {"split": "val (from official train)",
                 "policy": f"largest coverage with validation selective accuracy >= {TARGET_SELECTIVE_ACC:.0%} "
                           f"({'hybrid' if use_hybrid else 'deep-only'} calibrated probabilities)",
                 "curve": coverage_risk_curve(p_prod, y),
                 "summary": selective_summary(p_prod, y, final_th["threshold"]) if final_th["threshold"] is not None else None,
                 "crossfit_estimate": {k: float(np.mean([s[k] for s in cf_sel])) for k in ("coverage", "selective_accuracy", "rejection_rate")}
                 if cf_sel else None,
                 "coverage_at": {f"{c:.2f}": selective_summary(p_prod, y, select_threshold(p_prod, y, min_coverage=c)["threshold"])
                                 for c in (0.5, 0.7, 0.8, 0.9, 1.0)}}
    for name, obj in (("calibration.json", cal), ("fusion.json", fusion), ("selective.json", selective)):
        (RESULTS / name).write_text(json.dumps(obj, indent=1, default=float))
    decision = {"selected_classifier_run": sel, "mode": fusion["production_mode"], "temperature": T,
                "classical_temperature": Tc, "fusion_alpha": alpha if use_hybrid else None,
                "reject_threshold": final_th["threshold"], "threshold_policy": selective["policy"],
                "fitted_on": "full validation subset (official train split)"}
    (artifact_root() / "decision.json").write_text(json.dumps(decision, indent=1, default=float))
    print(json.dumps(decision, indent=1, default=float))
    print(df[["run", "top1", "top5", "macro_f1", "ece_raw", "ece_cal_crossfit", "temperature"]].to_string(index=False))


if __name__ == "__main__":
    main()
