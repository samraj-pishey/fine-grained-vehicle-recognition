"""Primary workflow: upload -> validate -> detect -> select -> classify -> result / uncertainty."""

from __future__ import annotations

import hashlib
import time

import plotly.graph_objects as go
import streamlit as st

from ..inference.pipeline import PipelineError
from ..preprocessing.image_io import ImageValidationError, UploadLimits, load_image_bytes
from ..visualization.visuals import draw_detections
from .runtime import InferenceTimeout, Runtime

LOW_DET_CONF = 0.10  # exploratory re-run threshold shown only as "low-confidence candidates"


def _reset():
    for k in ("vr_key", "vr_img", "vr_dets", "vr_low", "vr_det_ms", "vr_sel", "vr_result", "vr_meta"):
        st.session_state.pop(k, None)
    st.session_state["vr_uploader_nonce"] = st.session_state.get("vr_uploader_nonce", 0) + 1


def top5_chart(result) -> go.Figure:
    labels = [f"{p.rank}. {p.name}" for p in result.top5][::-1]
    vals = [p.probability for p in result.top5][::-1]
    fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h", text=[f"{v:.1%}" for v in vals], textposition="auto",
                           marker_color=["#1d78c4"] * 4 + ["#e63946" if not result.rejected else "#f4a261"]))
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(range=[0, 1], tickformat=".0%"),
                      yaxis=dict(automargin=True))
    return fig


def render(rt: Runtime) -> None:
    st.subheader("Vehicle Recognition")
    st.caption("Upload a JPEG or PNG photo of a car (max 10 MB, 25 MP). The image is processed in memory and is not "
               "stored. Predictions cover the 196 Stanford Cars classes (mostly 1991–2012 models); vehicles outside "
               "this set will be mapped to the most similar known class or flagged as uncertain.")
    c_up, c_btn = st.columns([5, 1])
    with c_up:
        up = st.file_uploader("Vehicle image", type=["jpg", "jpeg", "png"], accept_multiple_files=False,
                              key=f"vr_up_{st.session_state.get('vr_uploader_nonce', 0)}", label_visibility="collapsed")
    with c_btn:
        st.button("Reset", on_click=_reset, width="stretch")
    if up is None:
        st.info("**Start here:** choose an image. The detector will find vehicles; if several are found you can "
                "pick which one to identify.")
        return
    data = up.getvalue()
    key = hashlib.sha256(data).hexdigest()
    if st.session_state.get("vr_key") != key:
        for k in ("vr_img", "vr_dets", "vr_low", "vr_sel", "vr_result", "vr_meta"):
            st.session_state.pop(k, None)
        st.session_state["vr_key"] = key
        with st.status("Validating image…", expanded=False) as s:
            try:
                li = load_image_bytes(data, UploadLimits())
            except ImageValidationError as e:
                s.update(label="Image rejected", state="error")
                st.session_state["vr_meta"] = {"error": e}
            else:
                st.session_state["vr_img"] = li.rgb
                st.session_state["vr_meta"] = {"format": li.format, "size": li.original_size,
                                               "exif_orientation": li.exif_orientation}
                s.update(label="Image validated", state="complete")
    meta = st.session_state.get("vr_meta", {})
    if "error" in meta:
        e = meta["error"]
        titles = {"too_large": "File too large", "too_many_pixels": "Image dimensions too large",
                  "unsupported_format": "Unsupported format", "corrupt": "Corrupt image", "empty": "Empty file",
                  "too_small": "Image too small"}
        st.error(f"**{titles.get(e.code, 'Invalid image')}.** {e.message}")
        return
    img = st.session_state["vr_img"]
    if rt.recognizer is None:
        st.warning("The model bundle is not available on this server, so recognition cannot run.")
        st.image(img, caption="Uploaded image (validated)", width="stretch")
        return

    # ---- detection ----
    if "vr_dets" not in st.session_state:
        with st.spinner("Detecting vehicles…"):
            try:
                dets, ms = rt.run(rt.recognizer.detect, img)
                low = []
                if not dets:
                    low = rt.run(lambda im: rt.recognizer.detector.detect(im, conf_thr=LOW_DET_CONF), img)
            except InferenceTimeout as e:
                st.error(f"**Inference timeout.** {e}")
                return
            except PipelineError as e:
                st.error(f"**Detection error.** {e.message}")
                return
        st.session_state.update(vr_dets=dets, vr_low=low, vr_det_ms=ms)
    dets, low = st.session_state["vr_dets"], st.session_state["vr_low"]
    thr = rt.recognizer.manifest.detector.confidence_threshold
    candidates = dets
    if not dets:
        if not low:
            st.warning("**No vehicle detected.** Try a clearer photo where the car is large and unobstructed.")
            st.image(img, caption="Uploaded image", width="stretch")
            return
        st.warning(f"**Low detector confidence.** No detection reached the validated threshold ({thr:.2f}). "
                   f"{len(low)} low-confidence candidate(s) are shown; results for them are unreliable.")
        candidates = low

    col_img, col_res = st.columns([1.15, 1])
    with col_img:
        if len(candidates) > 1:
            st.info(f"**{len(candidates)} vehicles detected.** Default selection = highest detection confidence; "
                    "choose another box below if needed.")
        opts = {d.index: f"#{d.index} — detection confidence {d.score:.2f}" for d in candidates}
        sel = st.radio("Vehicle to identify", list(opts), format_func=opts.get, horizontal=len(opts) <= 4,
                       key=f"vr_sel_radio_{st.session_state['vr_key'][:8]}") if len(candidates) > 1 else candidates[0].index
        st.image(draw_detections(img, candidates, selected=sel), width="stretch",
                 caption=f"Detections (numbered by confidence) · detector {st.session_state['vr_det_ms']:.0f} ms")
    det = next(d for d in candidates if d.index == sel)

    # ---- classification (cached per image + selection) ----
    res_key = (st.session_state["vr_key"], sel)
    if st.session_state.get("vr_result", (None, None))[0] != res_key:
        with st.spinner("Classifying selected vehicle…"):
            t = time.perf_counter()
            try:
                result = rt.run(rt.recognizer.classify, img, det.box)
            except InferenceTimeout as e:
                st.error(f"**Inference timeout.** {e}")
                return
            except PipelineError as e:
                st.error(f"**Classification error ({e.code}).** {e.message}")
                return
            except Exception:  # noqa: BLE001
                st.error("**Internal error.** The request could not be completed. No prediction is shown.")
                return
            result.timings_ms["total_classify_ms"] = (time.perf_counter() - t) * 1000
        st.session_state["vr_result"] = (res_key, result)
    result = st.session_state["vr_result"][1]
    st.session_state["vr_selected_det"] = det

    with col_res:
        top = result.top1
        low_det = not dets
        if result.rejected:
            st.markdown(
                f'<div class="vr-card"><span class="vr-badge vr-warn">UNCERTAIN — abstained</span>'
                f'<div class="vr-big">No definitive identification</div>'
                f'<div class="vr-muted">Calibrated confidence {result.confidence:.1%} is below the validation-selected '
                f'threshold {result.reject_threshold:.1%}. The alternatives below are shown for transparency only.</div></div>',
                unsafe_allow_html=True)
        else:
            badge = '<span class="vr-badge vr-warn">low detector confidence</span>' if low_det else \
                '<span class="vr-badge vr-ok">identified</span>'
            st.markdown(
                f'<div class="vr-card">{badge}<div class="vr-big">{top.make} {top.model}</div>'
                f'<div>Year: <b>{top.year}</b> · Make: <b>{top.make}</b></div>'
                f'<div class="vr-muted">Class #{top.class_id}: “{top.name}”</div></div>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Detection", f"{det.score:.2f}", help="Detection confidence: the detector's score that this box contains a vehicle.")
        m2.metric("Classification", f"{result.confidence:.1%}",
                  help="Classification confidence: temperature-calibrated probability of the top class on validation data. It is model "
                       "confidence, not a guarantee of real-world correctness.")
        m3.metric("Mode", "Hybrid" if result.mode == "hybrid" else "Deep", help="Deep = CNN only; Hybrid = CNN fused with the classical HOG/LBP/HSV branch.")
        st.markdown("**Top-5 alternatives**")
        st.plotly_chart(top5_chart(result), width="stretch", config={"displayModeBar": False})
        t = result.timings_ms
        st.caption(f"Timing: detector {st.session_state['vr_det_ms']:.0f} ms · crop {t.get('crop', 0):.0f} ms · "
                   f"classifier {t.get('classifier', 0):.0f} ms"
                   + (f" · classical {t['classical']:.0f} ms" if "classical" in t else "")
                   + f" · classify total {t.get('total_classify_ms', 0):.0f} ms (server-side, warm)")
    with st.expander("Selected ROI and box details"):
        from ..preprocessing.roi import crop_roi

        c1, c2 = st.columns([1, 2])
        c1.image(crop_roi(img, det.box, rt.recognizer.manifest.classifier.roi_pad_frac), caption="Classifier input ROI")
        c2.json({"box_xyxy_px": [round(v, 1) for v in det.box], "detection_confidence": round(det.score, 4),
                 "image_size_wh": list(img.shape[1::-1]), "format": meta.get("format"),
                 "exif_orientation_corrected": meta.get("exif_orientation", 1) != 1,
                 "top5": [{"class": p.name, "p": round(p.probability, 4)} for p in result.top5]})
