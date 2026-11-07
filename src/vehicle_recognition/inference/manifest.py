"""Deployment manifest: a single, validated description of the production model bundle.

Validation at startup (fail closed): schema, no placeholder values, every artefact present with matching
SHA-256, class mapping has exactly `class_count` entries with contiguous ids, decision parameters in range.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ManifestError(RuntimeError):
    pass


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


class Artefact(BaseModel):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int | None = None


class DetectorSpec(Artefact):
    architecture: str
    format: Literal["onnx"] = "onnx"
    input_size: int = Field(gt=0)
    confidence_threshold: float = Field(gt=0.0, lt=1.0)
    nms_iou_threshold: float = Field(gt=0.0, lt=1.0)
    min_box_frac: float = Field(ge=0.0, lt=1.0, default=0.0)
    max_detections: int = Field(gt=0, default=20)


class ClassifierSpec(Artefact):
    architecture: str
    format: Literal["onnx"] = "onnx"
    input_size: int = Field(gt=0)
    roi_pad_frac: float = Field(ge=0.0, le=0.5)
    class_count: int = 196
    class_mapping: str
    class_mapping_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    preprocessing: Literal["squash_resize_imagenet_norm"] = "squash_resize_imagenet_norm"
    precision: Literal["fp32", "int8"] = "fp32"


class ClassicalSpecM(Artefact):
    feature_config_version: int


class DecisionSpec(BaseModel):
    mode: Literal["deep_only", "hybrid"]
    temperature: float = Field(gt=0.0)
    reject_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    fusion_alpha: float | None = Field(default=None, ge=0.0, le=1.0)
    classical_temperature: float | None = Field(default=None, gt=0.0)
    threshold_policy: str

    @model_validator(mode="after")
    def _hybrid_needs_alpha(self):
        if self.mode == "hybrid" and (self.fusion_alpha is None or self.classical_temperature is None):
            raise ValueError("hybrid mode requires fusion_alpha and classical_temperature")
        return self


class Manifest(BaseModel):
    schema_version: Literal[1]
    model_bundle_id: str
    created_utc: str
    source_commit: str
    detector: DetectorSpec
    classifier: ClassifierSpec
    classical: ClassicalSpecM | None = None
    decision: DecisionSpec
    evidence: dict = Field(default_factory=dict)

    @field_validator("model_bundle_id", "source_commit")
    @classmethod
    def _no_placeholders(cls, v: str):
        if not v or "..." in v or v.lower() in {"todo", "tbd", "placeholder", "selected-and-recorded"}:
            raise ValueError("placeholder value in manifest")
        return v

    @model_validator(mode="after")
    def _hybrid_needs_classical(self):
        if self.decision.mode == "hybrid" and self.classical is None:
            raise ValueError("hybrid mode requires a classical artefact")
        return self


def load_manifest(path: str | Path, verify_hashes: bool = True) -> tuple[Manifest, Path]:
    path = Path(path)
    if not path.is_file():
        raise ManifestError(f"Model manifest not found: {path}")
    try:
        m = Manifest.model_validate(json.loads(path.read_text()))
    except Exception as e:  # noqa: BLE001
        raise ManifestError(f"Invalid model manifest: {e}") from e
    base = path.parent
    arts = [("detector", m.detector), ("classifier", m.classifier)] + ([("classical", m.classical)] if m.classical else [])
    for name, a in arts:
        p = base / a.path
        if not p.is_file():
            raise ManifestError(f"{name} artefact missing: {a.path}")
        if verify_hashes and sha256_file(p) != a.sha256:
            raise ManifestError(f"{name} artefact checksum mismatch: {a.path}")
    cm = base / m.classifier.class_mapping
    if not cm.is_file():
        raise ManifestError("class mapping missing")
    if verify_hashes and sha256_file(cm) != m.classifier.class_mapping_sha256:
        raise ManifestError("class mapping checksum mismatch")
    classes = json.loads(cm.read_text())["classes"]
    if len(classes) != m.classifier.class_count or [c["class_id"] for c in classes] != list(range(m.classifier.class_count)):
        raise ManifestError("class mapping is incomplete or not in canonical order")
    return m, base
