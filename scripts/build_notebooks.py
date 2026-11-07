"""Generate the Colab and Kaggle training notebooks from shared cell definitions (keeps them consistent).

The notebooks are thin: preflight -> storage -> pinned code checkout -> pinned deps -> data check ->
scripts/run_training_plan.py (resumable) -> summary. All logic lives in version-controlled scripts.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_URL = "https://github.com/samraj-pishey/fine-grained-vehicle-recognition.git"
ROOT = Path(__file__).resolve().parents[1]


def md(s: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)}


def code(s: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(keepends=True)}


PREFLIGHT = code('''
# 1. Preflight: runtime / GPU report
import platform, subprocess, shutil, sys
print(sys.version); print(platform.platform())
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout or "NO GPU VISIBLE")
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
assert torch.cuda.is_available(), "No GPU: Runtime > Change runtime type > GPU (free T4). Do NOT buy compute units."
print("free disk (GB):", shutil.disk_usage("/").free // 2**30)
''')

CHECKOUT = code(f'''
# 3. Code checkout at a pinned commit + pinned training dependencies
import os, subprocess
REPO_URL = "{REPO_URL}"
COMMIT = "main"   # replace with the exact commit hash for a frozen, reproducible run
if not os.path.exists(CODE):
    subprocess.run(["git", "clone", "--quiet", REPO_URL, CODE], check=True)
subprocess.run(["git", "-C", CODE, "fetch", "--quiet", "origin"], check=True)
subprocess.run(["git", "-C", CODE, "checkout", "--quiet", COMMIT], check=True)
print("code @", subprocess.run(["git", "-C", CODE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout)
# Uses the platform's preinstalled CUDA build of torch/torchvision (versions are recorded in every run's env.json);
# everything else is pinned.
subprocess.run(["pip", "install", "-q", "-r", f"{{CODE}}/requirements/train-cloud.txt"], check=True)
''')

FETCH = code('''
# 3b. Data: download the ORIGINAL Stanford files directly (Internet Archive capture, SHA-256 verified) - no upload
#     needed. Builds train_val_annotations.csv from the official devkit + committed train/val split (no test data).
import os, subprocess, sys
if not (os.path.exists(ARCHIVE) and os.path.exists(TABLE)):
    subprocess.run([sys.executable, f"{CODE}/scripts/fetch_cloud_data.py", "--out", DATA_DIR], check=True)
''')

DATA_CHECK = code('''
# 4. Data check (the training workflow only ever sees TRAIN/VAL data; the official test split stays on the Mac)
import os, hashlib
for p in (ARCHIVE, TABLE):
    assert os.path.exists(p), f"missing {p} - see the instructions at the top of this notebook"
    print(p, os.path.getsize(p) // 2**20, "MB")
import pandas as pd
t = pd.read_csv(TABLE); print(t["subset"].value_counts().to_dict())
assert set(t["subset"]) == {"train", "val"}, "table must contain only train/val rows"
''')

RUN = code('''
# 5. Run the experiment plan (idempotent: re-run this cell after any disconnect - it resumes)
#    ONLY = "" runs everything in configs/training_plan.yaml; or e.g. ONLY = "det_yolo26n,cls_mnv3l_224"
ONLY = ""
import subprocess, sys
cmd = [sys.executable, f"{CODE}/scripts/run_training_plan.py", "--archive", ARCHIVE, "--table", TABLE,
       "--work", WORK, "--out", OUT, "--workers", str(WORKERS)]
if ONLY: cmd += ["--only", ONLY]
subprocess.run(cmd, check=True)
''')

SUMMARY = code('''
# 6. Summary of completed runs (copy/download this folder back to the Mac: DATA_ROOT/../artifacts/runs)
import glob, json, pandas as pd
for f in sorted(glob.glob(f"{OUT}/detectors/*/detector_val_metrics.json")):
    m = json.load(open(f)); print(f.split("/")[-2], {k: round(m[k], 4) for k in ("mAP50", "mAP50_95", "precision", "recall")})
for f in sorted(glob.glob(f"{OUT}/classifiers/*/history.csv")):
    h = pd.read_csv(f); b = h.loc[h.val_top1.idxmax()]
    print(f.split("/")[-2], "best epoch", int(b.epoch), "val top1 %.4f top5 %.4f" % (b.val_top1, b.val_top5))
''')

CLEANUP = md('''
## 7. Cleanup
* Ephemeral disk (`WORK`) is discarded automatically when the session ends.
* **Do not delete** anything under `OUT` until the runs have been copied to the Mac and their SHA-256 values
  checked against each run's `checkpoint_manifest.json` / `detector_val_metrics.json`.
* Disconnect the runtime when finished so free GPU quota is not wasted.
''')


def colab() -> dict:
    cells = [
        md('''
# Vehicle Recognition — GPU training (Google Colab, free tier)

**Before running** (one-time):
1. Nothing to upload: the notebook downloads the original Stanford files itself (SHA-256 verified) and caches
   them in `MyDrive/vehicle_recognition/data/` (~0.94 GB).
2. *Runtime → Change runtime type → T4 GPU* (free). **Never buy compute units / Colab Pro** (zero-cost project).
3. *Runtime → Run all*. Approve the Google Drive permission prompt yourself (never paste tokens anywhere).

If the session disconnects: reconnect and *Run all* again — completed stages are skipped and the interrupted
one resumes from its last verified checkpoint on Drive. Expected Drive usage: ≈1 GB data + ≈0.6 GB checkpoints.
'''),
        PREFLIGHT,
        code('''
# 2. Persistent storage: Google Drive (checkpoints/logs) + fast local disk (extracted images, caches)
from google.colab import drive
drive.mount("/content/drive")
BASE = "/content/drive/MyDrive/vehicle_recognition"
DATA_DIR = f"{BASE}/data"         # archive cached on Drive (~0.94 GB) so later sessions skip the download
ARCHIVE = f"{DATA_DIR}/cars_train.tgz"
TABLE = f"{DATA_DIR}/train_val_annotations.csv"
OUT = f"{BASE}/runs"            # persistent: survives disconnects
WORK = "/content/work"           # ephemeral: regenerated automatically
CODE = "/content/code"
WORKERS = 2                      # Colab free VMs have 2 vCPUs
import shutil; print("Drive free (GB):", shutil.disk_usage("/content/drive").free // 2**30)
'''),
        CHECKOUT, FETCH, DATA_CHECK, RUN, SUMMARY, CLEANUP,
    ]
    return nb(cells, colab=True)


def kaggle() -> dict:
    cells = [
        md('''
# Vehicle Recognition — GPU training (Kaggle Notebooks, free tier)

**Before running** (one-time):
1. Nothing to upload: the notebook downloads the original Stanford files itself (SHA-256 verified, ~10-20 min).
2. *Settings → Accelerator → GPU T4 x2 or P100* (free weekly quota),
   *Settings → Internet → On* (needs a phone-verified Kaggle account; used for `git clone` and `pip`).
3. Click **Save Version → Save & Run All (Commit)**: runs in the background (no browser needed, ≤ 12 h).
4. When finished, open the version's *Output* tab and download `runs/` to the Mac.

To resume after a timeout: add the previous version's output as an input and set `RESUME_FROM` below.
'''),
        PREFLIGHT,
        code('''
# 2. Storage: data downloaded to local disk + /kaggle/working (persisted as the notebook version output)
import os, shutil
DATA_DIR = "/kaggle/tmp/data"     # downloaded automatically in step 3b (no upload needed)
ARCHIVE = f"{DATA_DIR}/cars_train.tgz"
TABLE = f"{DATA_DIR}/train_val_annotations.csv"
OUT = "/kaggle/working/runs"
WORK = "/kaggle/tmp/work"
CODE = "/kaggle/tmp/code"
WORKERS = 4
RESUME_FROM = ""   # e.g. "/kaggle/input/<previous-notebook-output>/runs"
if RESUME_FROM and not os.path.exists(OUT):
    shutil.copytree(RESUME_FROM, OUT)
os.makedirs(DATA_DIR, exist_ok=True)
'''),
        CHECKOUT, FETCH, DATA_CHECK, RUN, SUMMARY, CLEANUP,
    ]
    return nb(cells, colab=False)


def nb(cells, colab: bool) -> dict:
    meta = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}}
    if colab:
        meta["accelerator"] = "GPU"
        meta["colab"] = {"provenance": [], "gpuType": "T4"}
    return {"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    for sub, fn in (("colab", colab), ("kaggle", kaggle)):
        p = ROOT / "notebooks" / sub / "train_vehicle_recognition.ipynb"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(fn(), indent=1) + "\n")
        print("wrote", p)
