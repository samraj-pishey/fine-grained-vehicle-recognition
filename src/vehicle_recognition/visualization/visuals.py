"""Image-processing / feature visualisations (these show processing steps and features, NOT CNN reasoning)."""

from __future__ import annotations

import cv2
import numpy as np

from ..classical.features import FeatureConfig, hog_features, lbp_map, prepare_roi, to_gray

PALETTE = [(230, 57, 70), (29, 120, 196), (42, 157, 143), (244, 162, 97), (131, 56, 236), (255, 190, 11)]


def draw_detections(img_rgb: np.ndarray, dets, selected: int | None = None) -> np.ndarray:
    out = img_rgb.copy()
    h, w = out.shape[:2]
    lw = max(2, round(min(h, w) / 220))
    fs = max(0.5, min(h, w) / 900)
    for d in dets:
        col = PALETTE[(d.index - 1) % len(PALETTE)]
        x0, y0, x1, y1 = (int(round(v)) for v in d.box)
        thick = lw * 2 if selected == d.index else lw
        cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), col, thick)
        label = f"#{d.index}  {d.score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, max(1, lw // 2))
        ty = max(th + 6, y0)
        cv2.rectangle(out, (x0, ty - th - 6), (x0 + tw + 8, ty), col, -1)
        cv2.putText(out, label, (x0 + 4, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), max(1, lw // 2), cv2.LINE_AA)
    return out


def grayscale(roi: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)


def canny(roi: np.ndarray, lo: int = 80, hi: int = 180) -> np.ndarray:
    return cv2.Canny(cv2.GaussianBlur(grayscale(roi), (5, 5), 1.0), lo, hi)


def sobel_magnitude(roi: np.ndarray) -> np.ndarray:
    g = grayscale(roi).astype(np.float32)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    return (255 * mag / max(1e-6, mag.max())).astype(np.uint8)


def edge_density(roi: np.ndarray) -> float:
    return float((canny(roi) > 0).mean())


def hog_image(roi: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    gray = to_gray(prepare_roi(roi, cfg), cfg)
    _, vis = hog_features(gray, cfg, visualize=True)
    vis = vis / max(1e-6, vis.max())
    return (255 * np.sqrt(vis)).astype(np.uint8)


def lbp_image(roi: np.ndarray, cfg: FeatureConfig) -> tuple[np.ndarray, np.ndarray]:
    gray = to_gray(prepare_roi(roi, cfg), cfg)
    codes = lbp_map(gray, cfg)
    nb = cfg.lbp_points + 2
    hist, _ = np.histogram(codes, bins=nb, range=(0, nb))
    return (255 * codes / max(1, nb - 1)).astype(np.uint8), hist / max(1, hist.sum())


def hsv_channels(roi: np.ndarray) -> dict[str, np.ndarray]:
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    return {"Hue": (hsv[..., 0].astype(np.float32) * 255 / 179).astype(np.uint8), "Saturation": hsv[..., 1], "Value": hsv[..., 2]}


def hsv_histograms(roi: np.ndarray, bins: int = 32) -> dict[str, np.ndarray]:
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    rng = {"Hue": 180, "Saturation": 256, "Value": 256}
    return {k: cv2.calcHist([hsv], [i], None, [bins], [0, rng[k]]).ravel() / hsv[..., 0].size for i, k in enumerate(rng)}


def letterbox_preview(img_rgb: np.ndarray, size: int) -> np.ndarray:
    from ..detection.onnx_detector import letterbox

    return letterbox(img_rgb, size)[0]
