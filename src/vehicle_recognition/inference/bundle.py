"""Model-bundle provisioning: if the deployment bundle is not present locally, download it once from a pinned,
public GitHub Release of this repository (configs/deployment.yaml) and verify the SHA-256 of the manifest (pinned
in config) and of every file it lists. No credentials are needed or used (public release, anonymous HTTPS)."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path

from ..config import load_yaml


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def _download(url: str, dst: Path, timeout: int = 120) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dst.parent, delete=False) as tmp:
        with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 - fixed https URL from config
            while chunk := r.read(1 << 20):
                tmp.write(chunk)
    Path(tmp.name).replace(dst)


def ensure_bundle(manifest_path: Path) -> None:
    if manifest_path.is_file():
        return
    if os.environ.get("MODEL_BUNDLE_DOWNLOAD", "1") == "0":
        raise FileNotFoundError(f"No local model bundle at {manifest_path} (download disabled by MODEL_BUNDLE_DOWNLOAD=0)")
    cfg = load_yaml("deployment.yaml").get("model_release") or {}
    repo, tag = cfg.get("github_repo"), cfg.get("tag")
    if not repo or not tag:
        raise FileNotFoundError(f"No local model bundle at {manifest_path} and no model release configured")
    base = f"https://github.com/{repo}/releases/download/{tag}/"
    bundle_dir = manifest_path.parent
    _download(base + "manifest.json", manifest_path)
    if cfg.get("manifest_sha256") and _sha256(manifest_path) != cfg["manifest_sha256"]:
        manifest_path.unlink(missing_ok=True)
        raise RuntimeError("Downloaded manifest does not match the pinned manifest_sha256")
    m = json.loads(manifest_path.read_text())
    files = [(m["detector"]["path"], m["detector"]["sha256"]), (m["classifier"]["path"], m["classifier"]["sha256"]),
             (m["classifier"]["class_mapping"], m["classifier"]["class_mapping_sha256"])]
    if m.get("classical"):
        files.append((m["classical"]["path"], m["classical"]["sha256"]))
    for rel, sha in files:
        dst = bundle_dir / rel
        if dst.is_file() and _sha256(dst) == sha:
            continue
        _download(base + rel, dst)
        if _sha256(dst) != sha:
            dst.unlink(missing_ok=True)
            manifest_path.unlink(missing_ok=True)
            raise RuntimeError(f"Checksum mismatch for downloaded {rel}")
