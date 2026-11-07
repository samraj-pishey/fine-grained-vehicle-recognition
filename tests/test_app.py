"""App state tests via Streamlit's headless AppTest."""

import os

import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_app_renders_without_model_bundle(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_MANIFEST_PATH", str(tmp_path / "missing" / "manifest.json"))
    monkeypatch.setenv("MODEL_BUNDLE_DOWNLOAD", "0")  # exercise the 'model unavailable' state, not the auto-download
    st.cache_resource.clear()
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=60).run()
    assert not at.exception, at.exception
    assert any("Model unavailable" in e.value for e in at.error)
    assert len(at.tabs) == 7
    # research tabs fall back to NOT RUN instead of placeholder numbers when artefacts are absent
    md = " ".join(m.value for m in at.markdown)
    assert "NOT RUN" in md or os.path.exists(os.path.join(ROOT, "reports", "results"))
