"""Controlled synthetic perturbations for robustness analysis (six conditions x severities 1..3).

Applied to the *full image* before cropping when the condition concerns capture (lighting, resolution, blur,
noise/compression) and to the ROI for occlusion/viewpoint. Deterministic given (seed, image index).
These are synthetic proxies, not natural conditions, and are never used for training or tuning.
"""

from __future__ import annotations

import cv2
import numpy as np

CONDITIONS = ("lighting", "occlusion", "viewpoint", "low_resolution", "blur", "noise_compression")
SEVERITIES = (1, 2, 3)


def lighting(img: np.ndarray, s: int, rng: np.random.Generator) -> np.ndarray:
    """Global gamma/brightness change plus a soft directional shadow."""
    gamma = {1: 1.4, 2: 1.9, 3: 2.6}[s] if rng.random() < 0.5 else {1: 0.75, 2: 0.55, 3: 0.4}[s]
    out = ((img / 255.0) ** gamma) * 255.0
    h, w = img.shape[:2]
    ang = rng.uniform(0, 2 * np.pi)
    yy, xx = np.mgrid[0:h, 0:w]
    ramp = ((xx - w / 2) * np.cos(ang) + (yy - h / 2) * np.sin(ang)) / max(h, w)
    shadow = 1.0 - {1: 0.25, 2: 0.4, 3: 0.55}[s] * (ramp > rng.uniform(-0.2, 0.2))
    out = out * cv2.GaussianBlur(shadow.astype(np.float32), (0, 0), max(h, w) / 40)[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def occlusion(roi: np.ndarray, s: int, rng: np.random.Generator) -> np.ndarray:
    """Grey rectangle covering 10/20/30 % of the ROI area at a random position."""
    out = roi.copy()
    h, w = roi.shape[:2]
    frac = {1: 0.10, 2: 0.20, 3: 0.30}[s]
    ar = rng.uniform(0.5, 2.0)
    oh = int(min(h, np.sqrt(frac * h * w / ar)))
    ow = int(min(w, frac * h * w / max(1, oh)))
    y0, x0 = rng.integers(0, max(1, h - oh + 1)), rng.integers(0, max(1, w - ow + 1))
    out[y0 : y0 + oh, x0 : x0 + ow] = 127
    return out


def viewpoint(roi: np.ndarray, s: int, rng: np.random.Generator) -> np.ndarray:
    """Perspective warp approximating an off-axis viewpoint change (corner displacement 8/15/22 %)."""
    h, w = roi.shape[:2]
    d = {1: 0.08, 2: 0.15, 3: 0.22}[s]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jit = rng.uniform(0, d, (4, 2)) * np.array([w, h])
    sign = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    dst = (src + sign * jit).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(roi, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def low_resolution(img: np.ndarray, s: int, rng: np.random.Generator) -> np.ndarray:
    """Downsample by 2/4/8 then upsample back (information loss at the same pixel size)."""
    f = {1: 2, 2: 4, 3: 8}[s]
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(8, w // f), max(8, h // f)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def blur(img: np.ndarray, s: int, rng: np.random.Generator) -> np.ndarray:
    """Gaussian (defocus) or linear motion blur, kernel scaled to image size."""
    k = max(3, int(round(min(img.shape[:2]) * {1: 0.01, 2: 0.02, 3: 0.035}[s])) | 1)
    if rng.random() < 0.5:
        return cv2.GaussianBlur(img, (k, k), 0)
    ker = np.zeros((k, k), np.float32)
    ker[k // 2, :] = 1.0 / k
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1.0)
    ker = cv2.warpAffine(ker, M, (k, k))
    return cv2.filter2D(img, -1, ker / max(ker.sum(), 1e-6))


def noise_compression(img: np.ndarray, s: int, rng: np.random.Generator) -> np.ndarray:
    """Gaussian sensor noise (sigma 8/16/28) followed by JPEG re-encoding (quality 50/25/10)."""
    sigma = {1: 8, 2: 16, 3: 28}[s]
    q = {1: 50, 2: 25, 3: 10}[s]
    noisy = np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(noisy, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


IMAGE_LEVEL = {"lighting": lighting, "low_resolution": low_resolution, "blur": blur, "noise_compression": noise_compression}
ROI_LEVEL = {"occlusion": occlusion, "viewpoint": viewpoint}


def perturb(img: np.ndarray, roi_box, condition: str, severity: int, seed: int, pad: float):
    """Return the perturbed ROI (H x W x 3 uint8) for a condition/severity."""
    from ..preprocessing.roi import crop_roi

    rng = np.random.default_rng(seed)
    if condition in IMAGE_LEVEL:
        return crop_roi(IMAGE_LEVEL[condition](img, severity, rng), roi_box, pad)
    if condition in ROI_LEVEL:
        return ROI_LEVEL[condition](crop_roi(img, roi_box, pad), severity, rng)
    raise ValueError(condition)
