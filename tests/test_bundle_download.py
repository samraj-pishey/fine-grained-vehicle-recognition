"""Bundle provisioning: downloads are checksum-verified and tampering is rejected (no network: _download mocked)."""
import json
import shutil

import pytest

from vehicle_recognition.inference import bundle as B
from vehicle_recognition.inference.manifest import load_manifest

from .test_detection_manifest_split import _write_bundle


def _serve(monkeypatch, src_dir, manifest_sha):
    monkeypatch.setattr(B, "load_yaml", lambda _: {"model_release": {"github_repo": "o/r", "tag": "t",
                                                                     "manifest_sha256": manifest_sha}})

    def fake_download(url, dst, timeout=120):
        assert url.startswith("https://github.com/o/r/releases/download/t/")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src_dir / url.rsplit("/", 1)[1], dst)

    monkeypatch.setattr(B, "_download", fake_download)


def test_download_and_verify(tmp_path, monkeypatch):
    src = tmp_path / "release"
    src.mkdir()
    p, _ = _write_bundle(src)
    _serve(monkeypatch, src, B._sha256(p))
    dst = tmp_path / "deploy" / "manifest.json"
    B.ensure_bundle(dst)
    m, _ = load_manifest(dst)
    assert m.model_bundle_id == "bundle-test"


def test_tampered_manifest_rejected(tmp_path, monkeypatch):
    src = tmp_path / "release"
    src.mkdir()
    _write_bundle(src)
    _serve(monkeypatch, src, "0" * 64)
    with pytest.raises(RuntimeError, match="manifest"):
        B.ensure_bundle(tmp_path / "deploy" / "manifest.json")


def test_tampered_file_rejected(tmp_path, monkeypatch):
    src = tmp_path / "release"
    src.mkdir()
    p, _ = _write_bundle(src)
    sha = B._sha256(p)
    (src / "cls.onnx").write_bytes(b"evil")
    _serve(monkeypatch, src, sha)
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        B.ensure_bundle(tmp_path / "deploy" / "manifest.json")
    assert not (tmp_path / "deploy" / "manifest.json").exists()
    _ = json
