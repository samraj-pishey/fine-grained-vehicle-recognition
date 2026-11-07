"""Configuration: resolve project paths from environment / .env / YAML. No user-specific hard-coding."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def _load_dotenv(path: Path = PROJECT_ROOT / ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines). Never overrides variables already set; never prints values."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


def load_yaml(name: str) -> dict[str, Any]:
    p = CONFIG_DIR / name if not name.startswith("/") else Path(name)
    with open(p) as f:
        return yaml.safe_load(f) or {}


@dataclass(frozen=True)
class DataPaths:
    root: Path

    @property
    def raw_archives(self) -> Path:
        return self.root / "raw_archives"

    @property
    def raw_extracted(self) -> Path:
        return self.root / "raw_extracted"

    @property
    def devkit(self) -> Path:
        return self.raw_extracted / "devkit"

    @property
    def test_labels_mat(self) -> Path:
        return self.raw_extracted / "test_labels" / "cars_test_annos_withlabels.mat"

    @property
    def train_images(self) -> Path:
        return self.raw_extracted / "cars_train"

    @property
    def test_images(self) -> Path:
        return self.raw_extracted / "cars_test"

    @property
    def manifests(self) -> Path:
        return self.root / "manifests"

    @property
    def metadata(self) -> Path:
        return self.root / "metadata"

    @property
    def processed(self) -> Path:
        return self.root / "processed"

    @property
    def roi_cache(self) -> Path:
        return self.root / "roi_cache"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    def image_path(self, split: str, fname: str) -> Path:
        return (self.train_images if split == "train" else self.test_images) / fname


def get_data_paths(data_root: str | os.PathLike | None = None) -> DataPaths:
    root = data_root or os.environ.get("DATA_ROOT")
    if not root:
        try:
            root = load_yaml("dataset.yaml").get("data_root")
        except FileNotFoundError:
            root = None
    if not root:
        raise RuntimeError(
            "DATA_ROOT is not configured. Set the DATA_ROOT environment variable, add it to .env, "
            "or set data_root in configs/dataset.yaml (see data/README.md)."
        )
    p = Path(os.path.expanduser(str(root)))
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return DataPaths(p.resolve())


def artifact_root() -> Path:
    return Path(os.environ.get("ARTIFACT_ROOT", PROJECT_ROOT / "artifacts")).resolve()
