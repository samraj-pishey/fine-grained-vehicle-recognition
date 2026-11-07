"""ONNX Runtime vehicle detector for Ultralytics YOLO exports (single 'vehicle' class).

Supports both output layouts, detected from the output shape:
  * end-to-end / NMS-free (YOLO26 default export): (1, N, 6) rows [x1, y1, x2, y2, score, cls] in input pixels
  * raw anchor-grid (YOLO11): (1, 4 + nc, A) rows [cx, cy, w, h, cls scores...] -> confidence filter + NMS here

Preprocessing reproduces Ultralytics LetterBox: aspect-preserving resize (bilinear), centred grey (114) padding,
RGB, /255, NCHW float32. Boxes are mapped back to original-image pixel coordinates and clipped.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Detection:
    box: tuple[float, float, float, float]  # x_min, y_min, x_max, y_max (0-based, half-open, original pixels)
    score: float
    index: int = 0  # 1-based display id after sorting

    def to_dict(self) -> dict:
        return {"id": self.index, "box": [round(v, 1) for v in self.box], "score": round(self.score, 4)}


def letterbox(img_rgb: np.ndarray, size: int, color: int = 114):
    h, w = img_rgb.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w, h) else img_rgb
    dw, dh = (size - nw) / 2, (size - nh) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(color, color, color))
    return out, r, (left, top)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    order = scores.argsort()[::-1]
    keep = []
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return np.array(keep, dtype=int)


def decode_output(out: np.ndarray, conf_thr: float, iou_thr: float, max_det: int = 50) -> tuple[np.ndarray, np.ndarray, str]:
    """Return (boxes_xyxy_in_letterbox_pixels, scores, layout)."""
    out = np.asarray(out)
    if out.ndim == 3:
        out = out[0]
    if out.ndim != 2:
        raise ValueError(f"Unexpected detector output shape {out.shape}")
    if out.shape[1] == 6 and out.shape[0] != 6:  # (N, 6) end-to-end
        scores = out[:, 4]
        m = scores >= conf_thr
        boxes, scores = out[m, :4].astype(np.float64), scores[m].astype(np.float64)
        # end-to-end models are trained one-to-one; a light NMS guards against rare duplicates
        if len(boxes):
            k = nms(boxes, scores, iou_thr)
            boxes, scores = boxes[k], scores[k]
        return boxes[:max_det], scores[:max_det], "end2end"
    # raw layout (4 + nc, A)
    pred = out if out.shape[0] < out.shape[1] else out.T
    xywh, cls = pred[:4].T, pred[4:].T
    scores = cls.max(axis=1)
    m = scores >= conf_thr
    xywh, scores = xywh[m].astype(np.float64), scores[m].astype(np.float64)
    boxes = np.stack([xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2,
                      xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2], axis=1) if len(xywh) else np.zeros((0, 4))
    if len(boxes):
        k = nms(boxes, scores, iou_thr)[:max_det]
        boxes, scores = boxes[k], scores[k]
    return boxes, scores, "raw_nms"


class OnnxDetector:
    def __init__(self, path: str, input_size: int = 640, conf_thr: float = 0.25, iou_thr: float = 0.7,
                 intra_threads: int = 2, inter_threads: int = 1, min_box_frac: float = 0.0):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = intra_threads
        so.inter_op_num_threads = inter_threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        shp = self.session.get_inputs()[0].shape
        self.input_size = int(shp[-1]) if isinstance(shp[-1], int) else input_size
        self.conf_thr, self.iou_thr, self.min_box_frac = conf_thr, iou_thr, min_box_frac
        self.layout: str | None = None

    def preprocess(self, img_rgb: np.ndarray):
        lb, r, pad = letterbox(img_rgb, self.input_size)
        x = np.ascontiguousarray(lb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
        return x, r, pad, lb

    def raw(self, x: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: x})[0]

    def detect(self, img_rgb: np.ndarray, conf_thr: float | None = None) -> list[Detection]:
        h, w = img_rgb.shape[:2]
        x, r, (pl, pt), _ = self.preprocess(img_rgb)
        boxes, scores, self.layout = decode_output(self.raw(x), self.conf_thr if conf_thr is None else conf_thr, self.iou_thr)
        dets = []
        for b, s in zip(boxes, scores, strict=True):
            x0 = float(np.clip((b[0] - pl) / r, 0, w))
            y0 = float(np.clip((b[1] - pt) / r, 0, h))
            x1 = float(np.clip((b[2] - pl) / r, 0, w))
            y1 = float(np.clip((b[3] - pt) / r, 0, h))
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            if (x1 - x0) * (y1 - y0) < self.min_box_frac * w * h:
                continue
            dets.append(Detection((x0, y0, x1, y1), float(s)))
        dets.sort(key=lambda d: -d.score)
        for i, d in enumerate(dets, 1):
            d.index = i
        return dets
