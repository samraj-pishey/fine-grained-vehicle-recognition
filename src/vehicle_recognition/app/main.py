"""Streamlit entrypoint: page layout, theming and tab routing. Business logic lives in inference/ and app/runtime."""

from __future__ import annotations

import logging

import streamlit as st

from . import tab_recognition, tab_research
from .runtime import Runtime

TITLE = "Intelligent Fine-Grained Vehicle Recognition and Analysis System"

CSS = """
<style>
  .block-container {padding-top: 1.6rem; max-width: 1280px;}
  h1, h2, h3 {font-family: "Source Serif 4", Georgia, serif; letter-spacing: -0.01em;}
  .vr-sub {opacity: .75; margin-top: -.6rem; font-size: .95rem;}
  .vr-card {border: 1px solid rgba(128,128,128,.28); border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: .8rem;}
  .vr-big {font-size: 1.65rem; font-weight: 650; line-height: 1.25;}
  .vr-muted {opacity: .7; font-size: .88rem;}
  .vr-badge {display:inline-block; padding:.12rem .55rem; border-radius:999px; font-size:.78rem; font-weight:600;
             border:1px solid rgba(128,128,128,.35); margin-right:.35rem;}
  .vr-ok {background: rgba(42,157,143,.15);} .vr-warn {background: rgba(244,162,97,.22);} .vr-bad {background: rgba(230,57,70,.18);}
  [data-testid="stMetricValue"] {font-size: 1.35rem;}
</style>
"""


@st.cache_resource(show_spinner="Loading models (first start after sleep can take a while)…")
def get_runtime() -> Runtime:
    return Runtime()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    st.set_page_config(page_title="Fine-Grained Vehicle Recognition", page_icon="🚗", layout="wide",
                       initial_sidebar_state="collapsed")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title(TITLE)
    st.markdown('<p class="vr-sub">Stanford Cars · 196 make/model/year classes · detector → ROI → calibrated classifier '
                '· research dashboard</p>', unsafe_allow_html=True)
    rt = get_runtime()
    if rt.error:
        st.error(f"**Model unavailable** ({rt.error[0]}). Recognition is disabled; research tabs still show recorded "
                 "results. Details are in the server log.")
    tabs = st.tabs(["Vehicle Recognition", "Detection & ROI Analysis", "Classical Computer Vision",
                    "Deep Learning & Hybrid Comparison", "Model Performance", "Robustness & Failure Analysis",
                    "About / Methodology"])
    with tabs[0]:
        tab_recognition.render(rt)
    with tabs[1]:
        tab_research.render_detection(rt)
    with tabs[2]:
        tab_research.render_classical(rt)
    with tabs[3]:
        tab_research.render_deep_hybrid(rt)
    with tabs[4]:
        tab_research.render_performance()
    with tabs[5]:
        tab_research.render_robustness()
    with tabs[6]:
        tab_research.render_about(rt)
