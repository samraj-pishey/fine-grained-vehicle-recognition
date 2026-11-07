"""End-to-end app tests through Streamlit's headless AppTest with real ONNX models (stand-in weights).

Covers: valid multi-vehicle image + user selection, low-confidence abstention, invalid file, corrupt file,
no vehicle, oversized file. Accuracy is meaningless here (random classifier); behaviour is what is tested."""

import io
import os

import numpy as np
import pytest
import streamlit as st
from PIL import Image
from streamlit.testing.v1 import AppTest

from .test_integration_pipeline import bundle, street_image  # noqa: F401  (fixtures)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pytestmark = pytest.mark.models


def _app(monkeypatch, bundle):  # noqa: F811
    monkeypatch.setenv("MODEL_MANIFEST_PATH", str(bundle / "manifest.json"))
    st.cache_resource.clear()
    return AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=120).run()


def _jpeg(arr) -> bytes:
    b = io.BytesIO()
    Image.fromarray(arr).save(b, format="JPEG", quality=90)
    return b.getvalue()


def _upload(at, name, data):
    at.file_uploader[0].upload(name, data, "image/jpeg").run()
    assert not at.exception, at.exception
    return at


def _text(at):
    return " ".join([m.value for m in at.markdown] + [e.value for e in at.error] + [w.value for w in at.warning]
                    + [i.value for i in at.info])


def test_multi_vehicle_selection_and_abstention(monkeypatch, bundle, street_image):  # noqa: F811
    at = _app(monkeypatch, bundle)
    assert not at.error
    _upload(at, "street.jpg", _jpeg(street_image))
    txt = _text(at)
    assert "vehicles detected" in txt  # multi-detection state
    assert len(at.radio) == 1 and len(at.radio[0].options) >= 2
    # random 196-way classifier -> confidence far below the 0.5 threshold -> must abstain, not assert a label
    assert "UNCERTAIN" in txt and "No definitive identification" in txt
    # user overrides the default selection
    at.radio[0].set_value(at.radio[0].options[1] if isinstance(at.radio[0].options[1], int) else 2).run()
    assert not at.exception, at.exception
    assert any(m.label == "Detection" for m in at.metric)


@pytest.mark.parametrize("name,data,expect", [
    ("notes.jpg", b"this is not an image at all" * 10, "Unsupported format"),
    ("big.jpg", b"\xff\xd8\xff" + b"0" * (11 * 1024 * 1024), "File too large"),
])
def test_invalid_inputs(monkeypatch, bundle, name, data, expect):  # noqa: F811
    at = _app(monkeypatch, bundle)
    _upload(at, name, data)
    assert any(expect in e.value for e in at.error), [e.value for e in at.error]
    assert "identified" not in _text(at)


def test_corrupt_image(monkeypatch, bundle, street_image):  # noqa: F811
    at = _app(monkeypatch, bundle)
    data = _jpeg(street_image)
    _upload(at, "broken.jpg", data[: len(data) // 4])
    assert any("Corrupt image" in e.value for e in at.error)


def test_no_vehicle(monkeypatch, bundle):  # noqa: F811
    at = _app(monkeypatch, bundle)
    _upload(at, "wall.jpg", _jpeg(np.full((480, 640, 3), 127, np.uint8)))
    assert any("No vehicle detected" in w.value for w in at.warning)
    assert not at.metric  # no fabricated result
