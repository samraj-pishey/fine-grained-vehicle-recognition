"""Research / analysis tabs. Every number shown comes from a committed experiment artefact in reports/results/;
anything missing is shown as 'NOT RUN'. Validation and official-test results are labelled separately."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ..classical.features import FeatureConfig
from ..preprocessing.roi import crop_roi
from ..visualization import visuals as V
from .runtime import Runtime, load_result


def not_run(what: str) -> None:
    st.markdown(f'<span class="vr-badge vr-bad">NOT RUN</span> {what} — no experiment artefact has been recorded.',
                unsafe_allow_html=True)


def split_badge(split: str) -> str:
    cls = "vr-warn" if "test" in split.lower() else "vr-ok"
    return f'<span class="vr-badge {cls}">{split}</span>'


def _show_table(res: dict | None, key: str, title: str, cols: list[str] | None = None, fmt: dict | None = None) -> pd.DataFrame | None:
    st.markdown(f"#### {title}")
    if not res or key not in res:
        not_run(title)
        return None
    st.markdown(split_badge(res.get("split", "?")) + f' <span class="vr-muted">{res.get("note", "")}</span>',
                unsafe_allow_html=True)
    df = pd.DataFrame(res[key])
    if cols:
        df = df[[c for c in cols if c in df.columns]]
    st.dataframe(df.style.format(fmt or {}, na_rep="—"), hide_index=True, width="stretch")
    return df


def _current_roi(rt: Runtime):
    img = st.session_state.get("vr_img")
    det = st.session_state.get("vr_selected_det")
    if img is None or det is None or rt.recognizer is None:
        return None, None, None
    return img, det, crop_roi(img, det.box, rt.recognizer.manifest.classifier.roi_pad_frac)


# ---------------------------------------------------------------- Detection & ROI
def render_detection(rt: Runtime) -> None:
    st.subheader("Detection & ROI Analysis")
    img, det, roi = _current_roi(rt)
    if img is None:
        st.info("Upload an image in **Vehicle Recognition** to see the detection pipeline for it.")
    else:
        dets = st.session_state.get("vr_dets") or st.session_state.get("vr_low") or []
        meta = st.session_state.get("vr_meta", {})
        size = rt.recognizer.manifest.detector.input_size
        c1, c2, c3 = st.columns(3)
        c1.image(img, caption=f"Orientation-corrected RGB input {img.shape[1]}×{img.shape[0]} "
                             f"(EXIF orientation tag {meta.get('exif_orientation', 1)})", width="stretch")
        c2.image(V.letterbox_preview(img, size), caption=f"Detector input: letterboxed {size}×{size}", width="stretch")
        c3.image(V.draw_detections(img, dets, det.index), caption="Boxes after confidence filter + NMS", width="stretch")
        c4, c5 = st.columns([1, 2])
        c4.image(roi, caption="Selected ROI (classifier input before resize)", width="stretch")
        c5.dataframe(pd.DataFrame([{"id": d.index, "confidence": round(d.score, 4),
                                    "x_min": round(d.box[0], 1), "y_min": round(d.box[1], 1),
                                    "x_max": round(d.box[2], 1), "y_max": round(d.box[3], 1),
                                    "width": round(d.box[2] - d.box[0], 1), "height": round(d.box[3] - d.box[1], 1),
                                    "area_frac": round((d.box[2] - d.box[0]) * (d.box[3] - d.box[1]) / (img.shape[0] * img.shape[1]), 3)}
                                   for d in dets]), hide_index=True, width="stretch")
    st.divider()
    res = load_result("detector_comparison.json")
    _show_table(res, "runs", "Detector comparison (YOLO26n vs YOLO11n)",
                fmt={c: "{:.4f}" for c in ["mAP50", "mAP50_95", "precision", "recall", "iou_mean", "iou_median", "miss_rate"]})
    if res and res.get("selected"):
        st.markdown(f"**Selected:** `{res['selected']}` — {res.get('rationale', '')}")
    iou = res.get("iou_hist") if res else None
    if iou:
        fig = go.Figure()
        for name, h in iou.items():
            fig.add_bar(x=h["bins"], y=h["counts"], name=name)
        fig.update_layout(title="IoU of best-matching prediction vs ground truth (validation)", barmode="group",
                          height=300, margin=dict(t=40, b=10), xaxis_title="IoU", yaxis_title="images")
        st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------- Classical CV
def render_classical(rt: Runtime) -> None:
    st.subheader("Classical Computer Vision")
    st.caption("These panels show image processing steps and hand-crafted features. They do not explain the CNN.")
    img, det, roi = _current_roi(rt)
    cfg = FeatureConfig()
    if rt.recognizer is not None and rt.recognizer.manifest.classical is not None and rt.recognizer.classical:
        cfg = rt.recognizer.classical["cfg"]
    if roi is None:
        st.info("Upload an image in **Vehicle Recognition** to visualise features of the selected vehicle.")
    else:
        r1 = st.columns(4)
        r1[0].image(roi, caption="ROI (RGB)", width="stretch")
        r1[1].image(V.grayscale(roi), caption="Grayscale", width="stretch")
        r1[2].image(V.canny(roi), caption=f"Canny edges (edge density {V.edge_density(roi):.3f})", width="stretch")
        r1[3].image(V.sobel_magnitude(roi), caption="Sobel gradient magnitude", width="stretch")
        r2 = st.columns(4)
        r2[0].image(V.hog_image(roi, cfg), caption=f"HOG ({cfg.hog_orientations} bins, {cfg.hog_pixels_per_cell[0]}px cells)",
                    width="stretch")
        lbp_img, lbp_hist = V.lbp_image(roi, cfg)
        r2[1].image(lbp_img, caption=f"LBP codes (P={cfg.lbp_points}, R={cfg.lbp_radius}, uniform)", width="stretch")
        fig = go.Figure(go.Bar(x=list(range(len(lbp_hist))), y=lbp_hist, marker_color="#2a9d8f"))
        fig.update_layout(title="LBP histogram", height=230, margin=dict(t=35, b=5, l=5, r=5))
        r2[2].plotly_chart(fig, width="stretch")
        ch = V.hsv_channels(roi)
        r2[3].image(ch["Hue"], caption="Hue channel", width="stretch")
        r3 = st.columns(3)
        r3[0].image(ch["Saturation"], caption="Saturation channel", width="stretch")
        r3[1].image(ch["Value"], caption="Value channel", width="stretch")
        hh = V.hsv_histograms(roi)
        fig = go.Figure([go.Scatter(y=v, name=k, mode="lines") for k, v in hh.items()])
        fig.update_layout(title="HSV histograms (32 bins)", height=230, margin=dict(t=35, b=5, l=5, r=5))
        r3[2].plotly_chart(fig, width="stretch")
        if rt.recognizer is not None and rt.recognizer.manifest.classical is not None:
            if st.button("Run the classical HOG+LBP+HSV → SVM classifier on this ROI"):
                with st.spinner("Loading classical model and extracting features…"):
                    rt.ensure_classical()
                    p = rt.run(rt.recognizer.classical_probs, roi)
                top = np.argsort(-p)[:5]
                st.dataframe(pd.DataFrame([{"rank": i + 1, "class": rt.recognizer.classes[c]["name"], "probability": float(p[c])}
                                           for i, c in enumerate(top)]).style.format({"probability": "{:.3f}"}),
                             hide_index=True, width="stretch")
    st.divider()
    res = load_result("classical_ablation.json")
    _show_table(res, "rows", "Feature-family ablation (calibrated RBF-SVM)",
                fmt={c: "{:.4f}" for c in ["top1", "top5", "macro_f1", "ece_15", "nll"]} | {"feat_ms": "{:.1f}", "svm_ms": "{:.1f}"})
    if res and res.get("pca_curve"):
        pc = res["pca_curve"]
        fig = go.Figure(go.Scatter(x=list(range(1, len(pc) + 1)), y=pc, mode="lines"))
        fig.update_layout(title="PCA cumulative explained variance (fit on training features only)", height=300,
                          xaxis_title="components", yaxis_title="cumulative variance", margin=dict(t=40, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption("Explained variance describes the feature space; it is not a measure of feature importance. "
                   "Importance is assessed by the family ablation above.")


# ---------------------------------------------------------------- Deep vs hybrid
def render_deep_hybrid(rt: Runtime) -> None:
    st.subheader("Deep Learning & Hybrid Comparison")
    img, det, roi = _current_roi(rt)
    res_state = st.session_state.get("vr_result")
    if roi is not None and res_state and rt.recognizer is not None and rt.recognizer.manifest.classical is not None:
        if st.button("Compare deep / classical / hybrid on the selected vehicle"):
            with st.spinner("Running all three branches…"):
                rt.ensure_classical()
                r = rt.run(rt.recognizer.classify, img, det.box, "hybrid")
            names = [c["name"] for c in rt.recognizer.classes]
            top = np.argsort(-r.probs)[:8]
            fig = go.Figure()
            for lab, p in (("Deep (calibrated)", r.deep_probs), ("Classical (calibrated)", r.classical_probs), ("Hybrid", r.probs)):
                fig.add_bar(x=[names[c] for c in top], y=p[top], name=lab)
            fig.update_layout(barmode="group", height=380, yaxis_tickformat=".0%", margin=dict(t=20, b=10))
            st.plotly_chart(fig, width="stretch")
            if rt.recognizer.manifest.decision.mode != "hybrid":
                st.caption("Production mode is deep-only; the hybrid output here is a research view with no validated "
                           "rejection threshold.")
    elif roi is None:
        st.info("Upload an image in **Vehicle Recognition** to compare branches on it.")
    st.divider()
    res = load_result("classifier_comparison.json")
    _show_table(res, "runs", "Deep classifier candidates",
                fmt={c: "{:.4f}" for c in ["top1", "top5", "macro_f1", "ece_raw", "ece_cal", "temperature", "worst_class_acc"]})
    if res and res.get("selected"):
        st.markdown(f"**Selected:** `{res['selected']}` — {res.get('rationale', '')}")
    fus = load_result("fusion.json")
    df = _show_table(fus, "comparison", "Deep-only vs classical vs hybrid",
                     fmt={c: "{:.4f}" for c in ["top1", "top5", "macro_f1", "ece_15", "nll", "brier"]})
    if fus and fus.get("grid"):
        g = pd.DataFrame(fus["grid"])
        fig = go.Figure([go.Scatter(x=g.alpha, y=g.top1, name="top-1", yaxis="y1"),
                         go.Scatter(x=g.alpha, y=g.nll, name="NLL", yaxis="y2")])
        fig.update_layout(title=f"Fusion weight α on validation (selected α = {fus.get('alpha')})", height=320,
                          xaxis_title="α (weight on deep branch)", yaxis=dict(title="top-1"),
                          yaxis2=dict(title="NLL", overlaying="y", side="right"), margin=dict(t=40, b=10))
        st.plotly_chart(fig, width="stretch")
        st.markdown(f"**Production decision:** {fus.get('production_mode', '—')} — {fus.get('rationale', '')}")
    _ = df


# ---------------------------------------------------------------- performance
def _reliability(fig, bins, name):
    b = [x for x in bins if x["count"]]
    fig.add_scatter(x=[x["confidence"] for x in b], y=[x["accuracy"] for x in b], mode="lines+markers", name=name)


def render_performance() -> None:
    st.subheader("Model Performance")
    cal = load_result("calibration.json")
    st.markdown("#### Confidence calibration")
    if not cal:
        not_run("Calibration analysis")
    else:
        st.markdown(split_badge(cal.get("split", "?")) + " ECE: 15 equal-width bins, top-label.", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(cal["table"]).style.format(precision=4), hide_index=True, width="stretch")
        fig = go.Figure()
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect", line=dict(dash="dash", color="gray"))
        for name, bins in cal.get("reliability", {}).items():
            _reliability(fig, bins, name)
        fig.update_layout(height=360, xaxis_title="confidence", yaxis_title="accuracy", margin=dict(t=20, b=10))
        st.plotly_chart(fig, width="stretch")
    sel = load_result("selective.json")
    st.markdown("#### Selective prediction (reject / abstain)")
    if not sel:
        not_run("Coverage–risk analysis")
    else:
        st.markdown(split_badge(sel.get("split", "?")) + f" Policy: {sel.get('policy', '')}", unsafe_allow_html=True)
        c = pd.DataFrame(sel["curve"]).dropna()
        fig = go.Figure(go.Scatter(x=c.coverage, y=c.selective_accuracy, mode="lines"))
        if sel.get("summary"):
            s = sel["summary"]
            fig.add_scatter(x=[s["coverage"]], y=[s["selective_accuracy"]], mode="markers", marker=dict(size=12),
                            name=f"selected threshold {s['threshold']:.3f}")
        fig.update_layout(height=320, xaxis_title="coverage", yaxis_title="selective accuracy", margin=dict(t=20, b=10))
        st.plotly_chart(fig, width="stretch")
        st.json(sel.get("summary", {}))
    _show_table(load_result("end_to_end.json"), "rows", "Evaluation tracks: GT-ROI, predicted-ROI and end-to-end")
    _show_table(load_result("benchmark.json"), "rows", "CPU latency, size and memory benchmark")
    ft = load_result("final_test.json")
    st.markdown("#### Official test set (evaluated once after configuration freeze)")
    if not ft:
        not_run("Final frozen test evaluation")
    else:
        st.markdown(split_badge("OFFICIAL TEST") + f' frozen commit `{ft.get("frozen_commit", "?")}`', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(ft["rows"]).style.format(precision=4), hide_index=True, width="stretch")


# ---------------------------------------------------------------- robustness
def render_robustness() -> None:
    st.subheader("Robustness & Failure Analysis")
    rob = load_result("robustness.json")
    st.markdown("#### Synthetic perturbation robustness")
    if not rob:
        not_run("Robustness evaluation")
    else:
        st.markdown(split_badge(rob.get("split", "?")) + " Synthetic perturbations (not natural conditions).",
                    unsafe_allow_html=True)
        df = pd.DataFrame(rob["rows"])
        fig = go.Figure()
        for cond, g in df.groupby("condition"):
            fig.add_scatter(x=g.severity, y=g.top1, mode="lines+markers", name=cond)
        fig.update_layout(height=360, xaxis_title="severity level", yaxis_title="top-1 accuracy", margin=dict(t=20, b=10))
        st.plotly_chart(fig, width="stretch")
        st.dataframe(df.style.format(precision=4), hide_index=True, width="stretch")
    fc = load_result("failure_cases.json")
    st.markdown("#### Documented failure cases")
    if not fc:
        not_run("Failure-case analysis")
    else:
        st.caption(f"{len(fc['cases'])} genuine case(s) from the {fc.get('split', '?')} split. Dataset images are not "
                   "shown because Stanford Cars may not be redistributed.")
        for c in fc["cases"]:
            with st.expander(f"{c['category']}: {c['image_id']} — true “{c['true_class']}” → predicted “{c['predicted_class']}”"):
                st.json(c)


# ---------------------------------------------------------------- about
def render_about(rt: Runtime) -> None:
    st.subheader("About / Methodology")
    m = rt.recognizer.manifest if rt.recognizer else None
    st.markdown(f"""
**Pipeline.** Uploaded images are validated by content (JPEG/PNG signatures, size and pixel limits, full decode),
EXIF-orientation corrected and converted to RGB. A fine-tuned single-class YOLO detector
({m.detector.architecture if m else '—'}, ONNX Runtime CPU) proposes vehicle boxes; the user selects one; the ROI
(with {m.classifier.roi_pad_frac if m else '—'} padding) is resized to
{m.classifier.input_size if m else '—'}² and classified by {m.classifier.architecture if m else '—'} into one of 196
Stanford Cars classes. Probabilities are temperature-calibrated on validation data and a validation-selected
threshold decides when to abstain. Make, model and year come from the official class names.

**Classical branch.** HOG + uniform LBP + HSV histograms → family-wise standardisation → (optional PCA) →
RBF-SVM with Platt calibration (5-fold cross-validated on training data). Hybrid fusion
P = α·P_deep + (1−α)·P_classical uses α selected on validation only.

**Scientific protocol.** Validation is drawn from the official training split (stratified, duplicate-grouped).
The official test split was used once, after all choices were frozen.

**Scope and limitations.** Recognises only the 196 Stanford Cars classes (mostly 1991–2012 US-market models).
It does not do number plates, tracking, damage, valuation, ownership or surveillance. Confidence is model
confidence. The detector was trained on single-car photos, so multi-vehicle scenes are supported in the
interface but not benchmarked.

**Privacy.** Images are processed in memory for the current session and are not stored or logged.

**Hosting.** Free Streamlit Community Cloud. After a period without traffic the app sleeps, and the first visit
must wake it and load the models, so a slow first response is expected.

**Data & licences.** Stanford Cars (Krause, Stark, Deng & Fei-Fei, 3dRR-13 @ ICCV 2013) is research-only.
No dataset images are redistributed. Model weights are for non-commercial research and education. The code is
AGPL-3.0 (Ultralytics YOLO). See THIRD_PARTY_NOTICES.md.
""")
    if m:
        st.markdown("#### Deployed model bundle")
        st.json({"bundle": m.model_bundle_id, "source_commit": m.source_commit, "detector": m.detector.architecture,
                 "classifier": f"{m.classifier.architecture} ({m.classifier.precision})", "mode": m.decision.mode,
                 "temperature": m.decision.temperature, "reject_threshold": m.decision.reject_threshold,
                 "fusion_alpha": m.decision.fusion_alpha})
