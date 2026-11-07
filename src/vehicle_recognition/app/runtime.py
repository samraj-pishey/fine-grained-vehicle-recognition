"""Process-level runtime services for the app: cached model bundle, bounded concurrency, timeouts, results."""

from __future__ import annotations

import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutTimeout
from pathlib import Path

from ..config import PROJECT_ROOT

log = logging.getLogger("vehicle_recognition.app")
RESULTS_DIR = PROJECT_ROOT / "reports" / "results"
DEFAULT_MANIFEST_PATH = str(PROJECT_ROOT / "artifacts" / "deployment" / "manifest.json")


def manifest_path_from_env() -> str:
    return os.environ.get("MODEL_MANIFEST_PATH", DEFAULT_MANIFEST_PATH)


INFERENCE_TIMEOUT_S = float(os.environ.get("INFERENCE_TIMEOUT_S", 60))
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT_INFERENCES", 1))


class InferenceTimeout(RuntimeError):
    pass


class Runtime:
    """One per process (held by st.cache_resource). Bounded concurrency: at most MAX_CONCURRENT inferences run;
    others wait up to the timeout, which keeps 2 vCPUs from being oversubscribed by simultaneous users."""

    def __init__(self, manifest_path: str | None = None):
        manifest_path = manifest_path or manifest_path_from_env()
        self.manifest_path = manifest_path
        self.error: tuple[str, str] | None = None
        self.recognizer = None
        self._sem = threading.BoundedSemaphore(MAX_CONCURRENT)
        self._pool = ThreadPoolExecutor(max_workers=MAX_CONCURRENT + 1, thread_name_prefix="infer")
        self._classical_lock = threading.Lock()
        try:
            from ..inference.bundle import ensure_bundle

            ensure_bundle(Path(manifest_path))
            from ..inference.pipeline import VehicleRecognizer

            self.recognizer = VehicleRecognizer(manifest_path, load_classical=None)
            log.info("model bundle loaded in %.0f ms", self.recognizer.load_ms)
        except Exception as e:  # noqa: BLE001 - surfaced as a 'model unavailable' state
            code = getattr(e, "code", "model_unavailable")
            self.error = (code, str(e))
            log.error("model bundle unavailable: %s", type(e).__name__)

    def run(self, fn, *args, timeout: float = INFERENCE_TIMEOUT_S):
        if not self._sem.acquire(timeout=timeout):
            raise InferenceTimeout("The server is busy; please retry in a moment.")
        try:
            fut = self._pool.submit(fn, *args)
            try:
                return fut.result(timeout=timeout)
            except FutTimeout as e:
                raise InferenceTimeout(f"Inference exceeded {timeout:.0f} s.") from e
        finally:
            self._sem.release()

    def ensure_classical(self) -> bool:
        """Lazily load the classical branch for research views (not loaded for deep-only production requests)."""
        r = self.recognizer
        if r is None or r.manifest.classical is None:
            return False
        if r.classical is None:
            with self._classical_lock:
                if r.classical is None:
                    from ..classical.features import FeatureConfig
                    from ..classical.model import load_model

                    bundle = load_model(Path(self.manifest_path).parent / r.manifest.classical.path)
                    r.classical = {
                        "pipe": bundle["pipeline"],
                        "cfg": FeatureConfig.from_dict(bundle["feature_config"]),
                        "families": tuple(bundle["spec"]["families"]),
                    }
        return True


def load_result(name: str):
    """Load a committed experiment artefact from reports/results; None means 'Not run'."""
    p = RESULTS_DIR / name
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text()) if p.suffix == ".json" else p.read_text()
    except Exception:  # noqa: BLE001
        return None
