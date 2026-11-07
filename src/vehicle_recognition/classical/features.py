"""Classical hand-crafted features: HOG, uniform LBP (spatial histograms) and HSV colour histograms.

All parameters live in a versioned FeatureConfig (configs/classical.yaml) that is saved with the model, so
train-time and inference-time extraction are identical.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

FAMILIES = ("hog", "lbp", "hsv")


@dataclass(frozen=True)
class FeatureConfig:
    version: int = 1
    resize_wh: tuple[int, int] = (160, 96)  # deterministic ROI resize (W, H); no letterbox
    interpolation: str = "area"
    # preprocessing (applied only where stated; validated by ablation)
    clahe: bool = False
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    gamma: float = 1.0
    # HOG
    hog_orientations: int = 9
    hog_pixels_per_cell: tuple[int, int] = (16, 16)
    hog_cells_per_block: tuple[int, int] = (2, 2)
    hog_block_norm: str = "L2-Hys"
    # LBP (uniform, rotation-variant "uniform" mapping has P+2 bins)
    lbp_points: int = 8
    lbp_radius: int = 1
    lbp_method: str = "uniform"
    lbp_grid: tuple[int, int] = (4, 4)  # spatial cells (rows, cols); per-cell normalised histograms
    # HSV joint histogram (OpenCV ranges H:[0,180), S,V:[0,256))
    hsv_bins: tuple[int, int, int] = (8, 4, 4)
    hsv_grid: tuple[int, int] = (1, 1)
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> FeatureConfig:
        d = dict(d)
        for k in ("resize_wh", "hog_pixels_per_cell", "hog_cells_per_block", "lbp_grid", "hsv_bins", "hsv_grid"):
            if k in d and d[k] is not None:
                d[k] = tuple(d[k])
        return FeatureConfig(**d)


_INTERP = {"area": cv2.INTER_AREA, "linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC}


def prepare_roi(roi_rgb: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    if roi_rgb.ndim != 3 or roi_rgb.shape[2] != 3 or roi_rgb.dtype != np.uint8:
        raise ValueError(f"Expected HxWx3 uint8 RGB, got {roi_rgb.shape} {roi_rgb.dtype}")
    img = cv2.resize(roi_rgb, cfg.resize_wh, interpolation=_INTERP[cfg.interpolation])
    if cfg.gamma != 1.0:
        lut = np.clip(((np.arange(256) / 255.0) ** (1.0 / cfg.gamma)) * 255.0, 0, 255).astype(np.uint8)
        img = cv2.LUT(img, lut)
    return img


def to_gray(img_rgb: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    g = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    if cfg.clahe:
        g = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=(cfg.clahe_grid, cfg.clahe_grid)).apply(g)
    return g


def hog_features(gray: np.ndarray, cfg: FeatureConfig, visualize: bool = False):
    return hog(
        gray,
        orientations=cfg.hog_orientations,
        pixels_per_cell=cfg.hog_pixels_per_cell,
        cells_per_block=cfg.hog_cells_per_block,
        block_norm=cfg.hog_block_norm,
        visualize=visualize,
        feature_vector=True,
    )


def lbp_map(gray: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    return local_binary_pattern(gray, cfg.lbp_points, cfg.lbp_radius, method=cfg.lbp_method)


def lbp_n_bins(cfg: FeatureConfig) -> int:
    if cfg.lbp_method == "uniform":
        return cfg.lbp_points + 2
    if cfg.lbp_method == "default":
        return 2**cfg.lbp_points
    raise ValueError(f"Unsupported LBP method {cfg.lbp_method}")


def _grid_slices(h: int, w: int, grid: tuple[int, int]):
    rs = np.linspace(0, h, grid[0] + 1).astype(int)
    cs = np.linspace(0, w, grid[1] + 1).astype(int)
    for i in range(grid[0]):
        for j in range(grid[1]):
            yield slice(rs[i], rs[i + 1]), slice(cs[j], cs[j + 1])


def lbp_features(gray: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    codes = lbp_map(gray, cfg)
    nb = lbp_n_bins(cfg)
    # Border policy: skimage computes LBP with interpolated neighbours; the outer `radius` ring is
    # excluded from histograms so that border behaviour is consistent across image sizes.
    r = cfg.lbp_radius
    codes = codes[r:-r, r:-r] if r > 0 else codes
    out = []
    for sy, sx in _grid_slices(*codes.shape, cfg.lbp_grid):
        hist, _ = np.histogram(codes[sy, sx], bins=nb, range=(0, nb))
        out.append(hist / max(1, hist.sum()))
    return np.concatenate(out).astype(np.float32)


def hsv_features(img_rgb: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    out = []
    for sy, sx in _grid_slices(*hsv.shape[:2], cfg.hsv_grid):
        cell = np.ascontiguousarray(hsv[sy, sx])
        hist = cv2.calcHist([cell], [0, 1, 2], None, list(cfg.hsv_bins), [0, 180, 0, 256, 0, 256]).reshape(-1)
        out.append(hist / max(1.0, hist.sum()))
    return np.concatenate(out).astype(np.float32)


def feature_dims(cfg: FeatureConfig) -> dict[str, int]:
    dummy = np.zeros((cfg.resize_wh[1], cfg.resize_wh[0], 3), np.uint8)
    f = extract_features(dummy, cfg)
    return {k: int(v.shape[0]) for k, v in f.items()}


def extract_features(roi_rgb: np.ndarray, cfg: FeatureConfig) -> dict[str, np.ndarray]:
    img = prepare_roi(roi_rgb, cfg)
    gray = to_gray(img, cfg)
    feats = {
        "hog": np.asarray(hog_features(gray, cfg), dtype=np.float32),
        "lbp": lbp_features(gray, cfg),
        "hsv": hsv_features(img, cfg),
    }
    for k, v in feats.items():
        if not np.all(np.isfinite(v)):
            raise ValueError(f"Non-finite values in {k} features")
    return feats
