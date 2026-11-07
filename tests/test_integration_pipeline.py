"""Integration: real ONNX detector + real ONNX classifier through VehicleRecognizer (manifest -> detect -> crop ->
classify -> calibrated Top-5 -> rejection). Uses the COCO-pretrained YOLO26n export as a stand-in detector and a
randomly initialised 196-class MobileNetV3 (so accuracy is meaningless; only behaviour is tested)."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vehicle_recognition.inference.manifest import sha256_file
from vehicle_recognition.inference.pipeline import PipelineError, VehicleRecognizer

ROOT = Path(__file__).resolve().parents[1]
DET = ROOT / "artifacts" / "pretrained" / "yolo26n.onnx"


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    torch = pytest.importorskip("torch")
    if not DET.exists():
        pytest.skip("pretrained detector export not available")
    from vehicle_recognition.classification.models import build_model, export_onnx

    d = tmp_path_factory.mktemp("bundle")
    torch.manual_seed(0)
    export_onnx(build_model("mobilenet_v3_large", 196, pretrained=False), str(d / "classifier.onnx"), 224)
    shutil.copy(DET, d / "detector.onnx")
    shutil.copy(ROOT / "reports" / "dataset" / "class_mapping.json", d / "class_mapping.json")
    m = {"schema_version": 1, "model_bundle_id": "integration-test", "created_utc": "2026-11-06T00:00:00Z",
         "source_commit": "test",
         "detector": {"architecture": "yolo26n-coco", "path": "detector.onnx", "sha256": sha256_file(d / "detector.onnx"),
                      "input_size": 640, "confidence_threshold": 0.25, "nms_iou_threshold": 0.7},
         "classifier": {"architecture": "mobilenet_v3_large", "path": "classifier.onnx",
                        "sha256": sha256_file(d / "classifier.onnx"), "input_size": 224, "roi_pad_frac": 0.05,
                        "class_mapping": "class_mapping.json", "class_mapping_sha256": sha256_file(d / "class_mapping.json")},
         "decision": {"mode": "deep_only", "temperature": 1.5, "reject_threshold": 0.5, "threshold_policy": "test"}}
    (d / "manifest.json").write_text(json.dumps(m))
    return d


@pytest.fixture(scope="module")
def street_image():
    from ultralytics.utils import ASSETS

    return np.asarray(Image.open(ASSETS / "bus.jpg").convert("RGB"))


@pytest.mark.models
def test_end_to_end_single_and_multi(bundle, street_image):
    r = VehicleRecognizer(str(bundle / "manifest.json"))
    dets, ms = r.detect(street_image)
    assert len(dets) >= 2 and ms > 0  # several objects -> multi-selection path
    assert [d.index for d in dets] == list(range(1, len(dets) + 1))
    assert all(dets[i].score >= dets[i + 1].score for i in range(len(dets) - 1))
    h, w = street_image.shape[:2]
    for d in dets:
        x0, y0, x1, y1 = d.box
        assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h
    res = r.classify(street_image, dets[1].box)  # user picks a non-default box
    assert len(res.top5) == 5 and abs(res.probs.sum() - 1) < 1e-6
    assert res.top1.name == json.loads((bundle / "class_mapping.json").read_text())["classes"][res.top1.class_id]["name"]
    assert res.rejected == (res.confidence < 0.5)


@pytest.mark.models
def test_no_vehicle_and_bad_box(bundle):
    r = VehicleRecognizer(str(bundle / "manifest.json"))
    blank = np.full((480, 640, 3), 127, np.uint8)
    dets, _ = r.detect(blank)
    assert dets == []
    with pytest.raises(PipelineError) as e:
        r.classify(blank, (10, 10, 10.5, 10.5))
    assert e.value.code == "bad_box"


@pytest.mark.models
def test_incompatible_classifier_size_fails_clearly(bundle, tmp_path):
    d = tmp_path / "b"
    shutil.copytree(bundle, d)
    m = json.loads((d / "manifest.json").read_text())
    m["classifier"]["input_size"] = 288  # ONNX graph is fixed at 224
    (d / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(PipelineError) as e:
        VehicleRecognizer(str(d / "manifest.json"))
    assert e.value.code == "model_incompatible"
