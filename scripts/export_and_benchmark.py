"""ONNX export, numerical parity, INT8 quantisation and CPU benchmarking.

  export-classifier : best.pt -> ONNX FP32; parity vs PyTorch on validation ROIs; INT8 (static QDQ, calibrated on
                      TRAIN ROIs only); accuracy/ECE of FP32 vs INT8 on validation.
  quantize-detector : detector ONNX -> INT8 static (calibrated on TRAIN images); box agreement vs FP32 on validation.
  bench             : latency P50/P95 (warm), cold load time, peak RSS, size - each config in a fresh subprocess.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def _rois(dp, subset: str, n: int, pad: float, seed: int = 0):
    from vehicle_recognition.preprocessing.image_io import load_dataset_image
    from vehicle_recognition.preprocessing.roi import crop_roi

    canon = pd.read_csv(dp.metadata / "canonical_annotations.csv")
    rows = canon[canon.subset == subset]
    if n and n < len(rows):
        rows = rows.sample(n=n, random_state=seed)
    for r in rows.itertuples(index=False):
        img = load_dataset_image(dp.raw_extracted / r.relative_path)
        yield crop_roi(img, (r.x_min, r.y_min, r.x_max, r.y_max), pad), int(r.class_id), img


def export_classifier(a):
    import onnxruntime as ort
    import torch
    from onnxruntime.quantization import (
        CalibrationDataReader,
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process
    from scipy.special import softmax

    from vehicle_recognition.calibration.calibration import ece
    from vehicle_recognition.classification.models import build_model, export_onnx
    from vehicle_recognition.classification.preprocess import preprocess_roi
    from vehicle_recognition.config import get_data_paths

    dp = get_data_paths()
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    size, pad = cfg["img_size"], cfg.get("roi_pad_frac", 0.05)
    # Read every image needed BEFORE exporting/quantising (decoding inside the quantiser's callback was observed to fail
    # after the torch export in the same process); inputs are kept as preprocessed arrays.
    calib_inputs = [preprocess_roi(r, size) for r, _, _ in _rois(dp, "train", a.calib_n, pad, seed=1)]
    parity_items = [(preprocess_roi(r, size), y) for r, y, _ in _rois(dp, "val", a.parity_n, pad)]
    eval_items = [(preprocess_roi(r, size), y) for r, y, _ in _rois(dp, "val", a.eval_n, pad, seed=2)]
    model = build_model(cfg["arch"], 196, pretrained=False)
    model.load_state_dict(ck["model"])
    model.eval()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fp32 = out / f"{a.name}_fp32.onnx"
    export_onnx(model, str(fp32), size)
    sess = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"])
    # parity on validation ROIs
    diffs, agree, X, Y = [], [], [], []
    for x, y in parity_items:
        with torch.no_grad():
            lt = model(torch.from_numpy(x)).numpy()[0]
        lo = sess.run(None, {"input": x})[0][0]
        diffs.append(float(np.abs(lt - lo).max()))
        agree.append(int(lt.argmax() == lo.argmax()))
        X.append(x)
        Y.append(y)

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter({"input": x} for x in calib_inputs)

        def get_next(self):
            return next(self.it, None)

    pre = out / f"{a.name}_fp32_pre.onnx"
    quant_pre_process(str(fp32), str(pre))
    int8 = out / f"{a.name}_int8{'' if a.calib_method == 'minmax' else '_' + a.calib_method}.onnx"
    method = {"minmax": CalibrationMethod.MinMax, "percentile": CalibrationMethod.Percentile,
              "entropy": CalibrationMethod.Entropy}[a.calib_method]
    extra = {"CalibPercentile": 99.99} if a.calib_method == "percentile" else {}
    quantize_static(str(pre), str(int8), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8, calibrate_method=method,
                    extra_options=extra)
    pre.unlink(missing_ok=True)
    s8 = ort.InferenceSession(str(int8), providers=["CPUExecutionProvider"])
    # FP32 vs INT8 on validation (full val subset for accuracy)
    L32, L8, Yv = [], [], []
    for x, y in eval_items:
        L32.append(sess.run(None, {"input": x})[0][0])
        L8.append(s8.run(None, {"input": x})[0][0])
        Yv.append(y)
    L32, L8, Yv = np.array(L32), np.array(L8), np.array(Yv)
    T = a.temperature
    rep = {"name": a.name, "arch": cfg["arch"], "img_size": size, "split": "val (from official train)",
           "parity": {"n": len(diffs), "max_abs_logit_diff": float(np.max(diffs)), "mean_max_abs_diff": float(np.mean(diffs)),
                      "top1_agreement": float(np.mean(agree))},
           "int8_calibration": {"method": f"static QDQ, per-channel weights, {a.calib_method}", "samples": a.calib_n, "source": "train subset ROIs"},
           "fp32": {"top1": float((L32.argmax(1) == Yv).mean()), "ece_15": ece(softmax(L32 / T, axis=1), Yv),
                    "bytes": fp32.stat().st_size, "sha256": sha256(fp32)},
           "int8": {"top1": float((L8.argmax(1) == Yv).mean()), "ece_15": ece(softmax(L8 / T, axis=1), Yv),
                    "bytes": int8.stat().st_size, "sha256": sha256(int8),
                    "top1_agreement_with_fp32": float((L8.argmax(1) == L32.argmax(1)).mean())},
           "eval_n": len(Yv), "temperature_used": T,
           "exporter": {"torch": torch.__version__, "onnxruntime": ort.__version__, "opset": 17}}
    (out / f"{a.name}_export_report_{a.calib_method}.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def quantize_detector(a):
    import onnxruntime as ort
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    from vehicle_recognition.config import get_data_paths
    from vehicle_recognition.detection.onnx_detector import OnnxDetector
    from vehicle_recognition.preprocessing.roi import iou

    dp = get_data_paths()
    src = Path(a.detector)
    d32 = OnnxDetector(str(src), conf_thr=a.conf)
    name = ort.InferenceSession(str(src)).get_inputs()[0].name

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = ({name: d32.preprocess(img)[0]} for _, _, img in _rois(dp, "train", a.calib_n, 0.0, seed=3))

        def get_next(self):
            return next(self.it, None)

    pre = src.with_name(src.stem + "_pre.onnx")
    quant_pre_process(str(src), str(pre))
    int8 = src.with_name(src.stem + "_int8.onnx")
    import onnx

    exclude = []
    if a.exclude_head:
        head = f"/model.{a.exclude_head}/"
        exclude = [n.name for n in onnx.load(str(pre)).graph.node if head in n.name]
    suffix = "_int8_headfp32" if exclude else "_int8"
    int8 = src.with_name(src.stem + suffix + ".onnx")
    quantize_static(str(pre), str(int8), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8, nodes_to_exclude=exclude)
    pre.unlink(missing_ok=True)
    d8 = OnnxDetector(str(int8), conf_thr=a.conf)
    ious, n32, n8 = [], 0, 0
    for _, _, img in _rois(dp, "val", a.eval_n, 0.0, seed=4):
        b32, b8 = d32.detect(img), d8.detect(img)
        n32 += bool(b32)
        n8 += bool(b8)
        if b32 and b8:
            ious.append(iou(b32[0].box, b8[0].box))
    rep = {"detector": src.name, "int8": int8.name, "eval_n": a.eval_n, "split": "val (from official train)",
           "excluded_nodes": len(exclude), "excluded_prefix": f"/model.{a.exclude_head}/" if exclude else None,
           "calibration": {"samples": a.calib_n, "source": "train subset images"},
           "fp32_images_with_det": n32, "int8_images_with_det": n8,
           "top_box_iou_fp32_vs_int8_mean": float(np.mean(ious)) if ious else None,
           "top_box_iou_fp32_vs_int8_p10": float(np.percentile(ious, 10)) if ious else None,
           "fp32_bytes": src.stat().st_size, "int8_bytes": int8.stat().st_size, "int8_sha256": sha256(int8)}
    (src.parent / f"{int8.stem}_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


BENCH_CHILD = r'''
import json, os, sys, time, resource, numpy as np
sys.path.insert(0, sys.argv[1])
kind, path, size, threads, n = sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
t0 = time.perf_counter()
if kind == "classifier":
    from vehicle_recognition.inference.pipeline import OnnxClassifier
    m = OnnxClassifier(path, size, intra_threads=threads, inter_threads=1)
    rng = np.random.default_rng(0); x = rng.integers(0, 255, (300, 520, 3), dtype=np.uint8)
    f = lambda: m.logits(x)
else:
    from vehicle_recognition.detection.onnx_detector import OnnxDetector
    m = OnnxDetector(path, size, 0.25, 0.7, intra_threads=threads, inter_threads=1)
    rng = np.random.default_rng(0); x = rng.integers(0, 255, (768, 1024, 3), dtype=np.uint8)
    f = lambda: m.detect(x)
load_ms = (time.perf_counter() - t0) * 1000
t = time.perf_counter(); f(); first_ms = (time.perf_counter() - t) * 1000
for _ in range(3): f()
ts = []
for _ in range(n):
    t = time.perf_counter(); f(); ts.append((time.perf_counter() - t) * 1000)
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
rss_mb = rss / 1024 / 1024 if sys.platform == "darwin" else rss / 1024
print(json.dumps({"load_ms": load_ms, "first_call_ms": first_ms, "p50_ms": float(np.percentile(ts, 50)),
                  "p95_ms": float(np.percentile(ts, 95)), "peak_rss_mb": rss_mb, "n": n}))
'''


def bench(a):
    import platform

    rows = []
    for spec in a.models:
        kind, path, size = spec.split(":")
        for th in [int(t) for t in a.threads.split(",")]:
            env = {**os.environ, "OMP_NUM_THREADS": str(th), "OPENBLAS_NUM_THREADS": "1"}
            r = subprocess.run([sys.executable, "-c", BENCH_CHILD, str(ROOT / "src"), kind, path, size, str(th), str(a.n)],
                               capture_output=True, text=True, env=env)
            if r.returncode != 0:
                print(r.stderr[-2000:])
                continue
            res = json.loads(r.stdout.strip().splitlines()[-1])
            rows.append({"kind": kind, "model": Path(path).name, "input": int(size), "threads": th,
                         "bytes": Path(path).stat().st_size, **res})
            print(rows[-1], flush=True)
    hw = {"machine": platform.machine(), "platform": platform.platform(), "processor": platform.processor(),
          "cpu_count": os.cpu_count(), "python": platform.python_version()}
    try:
        import onnxruntime

        hw["onnxruntime"] = onnxruntime.__version__
    except ImportError:
        pass
    out = {"split": "synthetic input (latency only)", "hardware": hw, "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "protocol": f"fresh process per row; 1 warm-up call (first_call_ms) + 3 discarded; P50/P95 over {a.n} calls; "
                       "batch 1; peak RSS of the whole process", "rows": rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export-classifier")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--name", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--temperature", type=float, default=1.0)
    e.add_argument("--parity-n", type=int, default=200)
    e.add_argument("--calib-n", type=int, default=300)
    e.add_argument("--eval-n", type=int, default=0, help="0 = full validation subset")
    e.add_argument("--calib-method", choices=["minmax", "percentile", "entropy"], default="minmax")
    q = sub.add_parser("quantize-detector")
    q.add_argument("--detector", required=True)
    q.add_argument("--conf", type=float, default=0.25)
    q.add_argument("--calib-n", type=int, default=200)
    q.add_argument("--eval-n", type=int, default=300)
    q.add_argument("--exclude-head", default=None, help="module index of the detection head to keep in float, e.g. 23")
    b = sub.add_parser("bench")
    b.add_argument("--models", nargs="+", required=True, help="kind:path:input_size")
    b.add_argument("--threads", default="1,2,4")
    b.add_argument("--n", type=int, default=50)
    b.add_argument("--out", required=True)
    a = ap.parse_args()
    {"export-classifier": export_classifier, "quantize-detector": quantize_detector, "bench": bench}[a.cmd](a)


if __name__ == "__main__":
    main()
