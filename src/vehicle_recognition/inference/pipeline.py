"""Production inference pipeline: detector -> (user-selected) ROI -> classifier [-> classical -> fusion]
-> temperature calibration -> rejection policy -> Top-5 + make/model/year.

Only the artefacts named in the deployment manifest are loaded. Every failure surfaces as a typed
PipelineError with a stable code; no failure path ever produces a vehicle label.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.special import softmax

from ..classification.preprocess import preprocess_roi
from ..detection.onnx_detector import Detection, OnnxDetector
from ..preprocessing.roi import crop_roi
from .manifest import Manifest, load_manifest

MIN_ROI_PX = 8  # smaller crops carry no usable fine-grained detail; reported as a 'bad_box' state


class PipelineError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class Prediction:
    rank: int
    class_id: int
    probability: float
    name: str
    make: str
    model: str
    year: int


@dataclass
class ClassificationResult:
    mode: str  # "deep_only" | "hybrid"
    top5: list[Prediction]
    confidence: float  # calibrated top-1 probability of the reported mode
    rejected: bool
    reject_threshold: float | None
    probs: np.ndarray  # calibrated probabilities (reported mode)
    deep_probs: np.ndarray
    classical_probs: np.ndarray | None
    timings_ms: dict = field(default_factory=dict)

    @property
    def top1(self) -> Prediction:
        return self.top5[0]


class OnnxClassifier:
    def __init__(self, path: str, input_size: int, intra_threads: int = 2, inter_threads: int = 1):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = intra_threads
        so.inter_op_num_threads = inter_threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size
        shp = self.session.get_inputs()[0].shape
        if isinstance(shp[-1], int) and shp[-1] != input_size:
            raise PipelineError("model_incompatible", f"classifier expects {shp[-1]}px input, manifest says {input_size}")

    def logits(self, roi_rgb: np.ndarray) -> np.ndarray:
        x = preprocess_roi(roi_rgb, self.input_size)
        return self.session.run(None, {self.input_name: x})[0][0].astype(np.float64)


class VehicleRecognizer:
    def __init__(self, manifest_path: str, load_classical: bool | None = None, verify_hashes: bool = True):
        intra = int(os.environ.get("ORT_INTRA_OP_THREADS", 2))
        inter = int(os.environ.get("ORT_INTER_OP_THREADS", 1))
        t0 = time.perf_counter()
        self.manifest, base = load_manifest(manifest_path, verify_hashes=verify_hashes)
        m: Manifest = self.manifest
        d = m.detector
        self.detector = OnnxDetector(str(base / d.path), d.input_size, d.confidence_threshold, d.nms_iou_threshold,
                                     intra, inter, d.min_box_frac)
        self.classifier = OnnxClassifier(str(base / m.classifier.path), m.classifier.input_size, intra, inter)
        self.classes = json.loads((base / m.classifier.class_mapping).read_text())["classes"]
        self.classical = None
        want_classical = (m.decision.mode == "hybrid") if load_classical is None else load_classical
        if want_classical and m.classical is not None:
            from ..classical.features import FeatureConfig
            from ..classical.model import load_model

            bundle = load_model(base / m.classical.path)
            self.classical = {"pipe": bundle["pipeline"], "cfg": FeatureConfig.from_dict(bundle["feature_config"]),
                              "families": tuple(bundle["spec"]["families"])}
        self.load_ms = (time.perf_counter() - t0) * 1000

    # ---------------- detection ----------------
    def detect(self, img_rgb: np.ndarray) -> tuple[list[Detection], float]:
        t = time.perf_counter()
        try:
            dets = self.detector.detect(img_rgb)
        except Exception as e:  # noqa: BLE001
            raise PipelineError("inference_error", "Vehicle detection failed.") from e
        return dets[: self.manifest.detector.max_detections], (time.perf_counter() - t) * 1000

    # ---------------- classification ----------------
    def classical_probs(self, roi_rgb: np.ndarray) -> np.ndarray:
        from ..classical.features import extract_features
        from ..classical.model import predict_proba_canonical, stack_families

        if self.classical is None:
            raise PipelineError("model_unavailable", "Classical branch is not loaded.")
        f = extract_features(roi_rgb, self.classical["cfg"])
        X, _ = stack_families({k: v[None] for k, v in f.items()}, self.classical["families"])
        return predict_proba_canonical(self.classical["pipe"], X, len(self.classes))[0]

    def classify(self, img_rgb: np.ndarray, box, mode: str | None = None) -> ClassificationResult:
        dec = self.manifest.decision
        mode = mode or dec.mode
        timings = {}
        t = time.perf_counter()
        try:
            roi = crop_roi(img_rgb, box, pad_frac=self.manifest.classifier.roi_pad_frac, min_size=MIN_ROI_PX)
        except ValueError as e:
            raise PipelineError("bad_box", "The selected box is too small to classify.") from e
        timings["crop"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        try:
            logits = self.classifier.logits(roi)
        except Exception as e:  # noqa: BLE001
            raise PipelineError("inference_error", "Classification failed.") from e
        timings["classifier"] = (time.perf_counter() - t) * 1000
        p_dl = softmax(logits / dec.temperature)
        p_cv = None
        if mode == "hybrid" or self.classical is not None:
            if self.classical is None:
                raise PipelineError("model_unavailable", "Hybrid mode requested but the classical branch is not loaded.")
            t = time.perf_counter()
            raw = self.classical_probs(roi)
            ct = dec.classical_temperature or 1.0
            p_cv = softmax(np.log(np.clip(raw, 1e-12, 1.0)) / ct)
            timings["classical"] = (time.perf_counter() - t) * 1000
        if mode == "hybrid":
            if dec.fusion_alpha is None:
                raise PipelineError("model_incompatible", "Hybrid mode has no validated fusion weight.")
            p = dec.fusion_alpha * p_dl + (1 - dec.fusion_alpha) * p_cv
            p = p / p.sum()
        else:
            p = p_dl
        if not (np.all(np.isfinite(p)) and abs(p.sum() - 1) < 1e-4 and p.shape[0] == len(self.classes)):
            raise PipelineError("inference_error", "Model produced an invalid probability vector.")
        order = np.argsort(-p)[:5]
        top5 = [Prediction(rank=i + 1, class_id=int(c), probability=float(p[c]), name=self.classes[c]["name"],
                           make=self.classes[c]["make"], model=self.classes[c]["model"], year=int(self.classes[c]["year"]))
                for i, c in enumerate(order)]
        conf = float(p[order[0]])
        thr = dec.reject_threshold if mode == dec.mode else None
        if mode != dec.mode:
            # thresholds are validated per mode; a non-production mode is shown without a definitive claim policy
            thr = None
        return ClassificationResult(mode=mode, top5=top5, confidence=conf, rejected=thr is not None and conf < thr,
                                    reject_threshold=thr, probs=p, deep_probs=p_dl, classical_probs=p_cv, timings_ms=timings)
