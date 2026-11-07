"""Deterministic classifier preprocessing shared by training-time evaluation and ONNX inference.

Policy: the (ground-truth or detected) vehicle ROI is resized *without* aspect preservation ("squash") to
S x S so the whole vehicle is kept (centre-cropping would cut off bumpers/lights that distinguish model years).
INTER_AREA when downscaling, INTER_LINEAR when upscaling; ImageNet mean/std normalisation; NCHW float32.
"""

from __future__ import annotations

import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resize_square(roi_rgb: np.ndarray, size: int) -> np.ndarray:
    h, w = roi_rgb.shape[:2]
    interp = cv2.INTER_AREA if (h > size or w > size) else cv2.INTER_LINEAR
    return cv2.resize(roi_rgb, (size, size), interpolation=interp)


def to_tensor_array(img_rgb_sq: np.ndarray) -> np.ndarray:
    x = (img_rgb_sq.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def preprocess_roi(roi_rgb: np.ndarray, size: int) -> np.ndarray:
    """Returns (1, 3, S, S) float32."""
    if roi_rgb.ndim != 3 or roi_rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB ROI, got {roi_rgb.shape}")
    return to_tensor_array(resize_square(roi_rgb, size))[None]
