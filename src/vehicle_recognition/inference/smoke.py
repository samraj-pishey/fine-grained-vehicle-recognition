"""In-container smoke test: `python -m vehicle_recognition.inference.smoke`.

With a deployment bundle: loads it (checksums verified), runs detect + classify on a synthetic image and checks
output invariants (it does NOT assert any accuracy). Without a bundle: verifies imports and that the runtime
reports 'model unavailable' instead of crashing. Exit code 0 = pass.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np


def main() -> int:
    from vehicle_recognition.app.runtime import Runtime

    t = time.perf_counter()
    rt = Runtime()
    load_ms = (time.perf_counter() - t) * 1000
    if rt.recognizer is None:
        print(json.dumps({"bundle": "absent", "state": rt.error[0] if rt.error else None, "load_ms": round(load_ms)}))
        return 0 if rt.error else 1
    img = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    dets, det_ms = rt.recognizer.detect(img)
    box = dets[0].box if dets else (100.0, 100.0, 500.0, 380.0)
    res = rt.recognizer.classify(img, box)
    assert len(res.top5) == 5 and abs(res.probs.sum() - 1) < 1e-4 and 0 <= res.confidence <= 1
    assert [p.probability for p in res.top5] == sorted((p.probability for p in res.top5), reverse=True)
    print(json.dumps({"bundle": rt.recognizer.manifest.model_bundle_id, "load_ms": round(load_ms), "det_ms": round(det_ms),
                      "n_det_on_noise": len(dets), "classify_ms": round(sum(res.timings_ms.values())),
                      "invariants": "ok"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
