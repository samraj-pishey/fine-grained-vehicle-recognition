import json

import numpy as np
import pandas as pd
import pytest

from vehicle_recognition.data.splits import duplicate_groups, make_train_val_split
from vehicle_recognition.detection.onnx_detector import decode_output, letterbox, nms
from vehicle_recognition.inference.manifest import ManifestError, load_manifest, sha256_file


def test_letterbox_geometry():
    img = np.zeros((300, 600, 3), np.uint8)
    out, r, (pl, pt) = letterbox(img, 640)
    assert out.shape == (640, 640, 3) and abs(r - 640 / 600) < 1e-9 and pl == 0 and pt == 160


def test_decode_end2end_layout():
    out = np.zeros((1, 300, 6), np.float32)
    out[0, 0] = [10, 20, 110, 220, 0.9, 0]
    out[0, 1] = [12, 22, 108, 218, 0.8, 0]   # duplicate -> suppressed by guard NMS
    out[0, 2] = [300, 300, 400, 380, 0.3, 0]
    boxes, scores, layout = decode_output(out, conf_thr=0.25, iou_thr=0.7)
    assert layout == "end2end" and len(boxes) == 2 and scores[0] == pytest.approx(0.9)


def test_decode_raw_layout_with_nms():
    A = 50
    out = np.zeros((1, 5, A), np.float32)
    out[0, :, 0] = [100, 100, 50, 40, 0.95]
    out[0, :, 1] = [101, 100, 50, 40, 0.90]   # overlaps -> removed
    out[0, :, 2] = [300, 300, 60, 60, 0.50]
    boxes, scores, layout = decode_output(out, conf_thr=0.25, iou_thr=0.7)
    assert layout == "raw_nms" and len(boxes) == 2
    assert np.allclose(boxes[0], [75, 80, 125, 120])


def test_nms_keeps_disjoint():
    b = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], float)
    assert set(nms(b, np.array([0.5, 0.6]), 0.5)) == {0, 1}


def test_split_is_grouped_stratified_and_deterministic():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"split": "train", "class_id": np.repeat(np.arange(10), 20)})
    pairs = [(0, 1), (1, 2)]  # duplicate cluster in class 0
    df["dup_group"] = duplicate_groups(len(df), pairs)
    a1 = make_train_val_split(df, 0.2, seed=5)
    a2 = make_train_val_split(df, 0.2, seed=5)
    assert a1.equals(a2)
    assert len(set(a1.iloc[[0, 1, 2]])) == 1  # cluster on one side
    per_class_val = (a1 == "val").groupby(df["class_id"]).sum()
    assert per_class_val.between(3, 6).all()
    with pytest.raises(ValueError):
        make_train_val_split(df.assign(split="test"), 0.2, seed=5)
    _ = rng


def _write_bundle(tmp_path, **over):
    (tmp_path / "det.onnx").write_bytes(b"det")
    (tmp_path / "cls.onnx").write_bytes(b"cls")
    classes = [{"class_id": i, "name": f"Make M{i} 2012", "make": "Make", "model": f"M{i}", "year": 2012} for i in range(196)]
    (tmp_path / "class_mapping.json").write_text(json.dumps({"classes": classes}))
    m = {"schema_version": 1, "model_bundle_id": "bundle-test", "created_utc": "2026-11-06T00:00:00Z",
         "source_commit": "abc123",
         "detector": {"architecture": "yolo26n", "path": "det.onnx", "sha256": sha256_file(tmp_path / "det.onnx"),
                      "input_size": 640, "confidence_threshold": 0.25, "nms_iou_threshold": 0.7},
         "classifier": {"architecture": "mobilenet_v3_large", "path": "cls.onnx", "sha256": sha256_file(tmp_path / "cls.onnx"),
                        "input_size": 224, "roi_pad_frac": 0.05, "class_mapping": "class_mapping.json",
                        "class_mapping_sha256": sha256_file(tmp_path / "class_mapping.json")},
         "decision": {"mode": "deep_only", "temperature": 1.3, "reject_threshold": 0.4, "threshold_policy": "test"}}
    for k, v in over.items():
        m[k] = v
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(m))
    return p, m


def test_manifest_valid(tmp_path):
    p, _ = _write_bundle(tmp_path)
    m, _ = load_manifest(p)
    assert m.decision.mode == "deep_only"


def test_manifest_rejects_tampered_artifact(tmp_path):
    p, _ = _write_bundle(tmp_path)
    (tmp_path / "cls.onnx").write_bytes(b"tampered")
    with pytest.raises(ManifestError, match="checksum"):
        load_manifest(p)


def test_manifest_rejects_placeholders_and_hybrid_without_classical(tmp_path):
    p, _ = _write_bundle(tmp_path, model_bundle_id="TODO")
    with pytest.raises(ManifestError):
        load_manifest(p)
    p, m = _write_bundle(tmp_path, decision={"mode": "hybrid", "temperature": 1.0, "fusion_alpha": 0.7,
                                             "classical_temperature": 1.0, "threshold_policy": "x"})
    with pytest.raises(ManifestError):
        load_manifest(p)


def test_manifest_missing(tmp_path):
    with pytest.raises(ManifestError, match="not found"):
        load_manifest(tmp_path / "nope.json")
