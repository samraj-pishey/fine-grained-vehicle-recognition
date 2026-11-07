"""Follow-up ablation (validation only): family weighting 'equal' (1/sqrt(dim)) vs 'none' for HOG+LBP+HSV,
motivated by the observation that HOG alone beat the equally-weighted combination. Appends rows to
reports/results/classical_ablation.json."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_classical import fit_eval  # noqa: E402

from vehicle_recognition.classical.model import ClassicalSpec  # noqa: E402
from vehicle_recognition.config import PROJECT_ROOT, get_data_paths  # noqa: E402

dp = get_data_paths()
tr = dict(np.load(dp.processed / "features" / "train_v1.npz"))
va = dict(np.load(dp.processed / "features" / "val_v1.npz"))
res = PROJECT_ROOT / "reports" / "results" / "classical_ablation.json"
d = json.loads(res.read_text())
for pca in (None, 0.95):
    spec = ClassicalSpec(families=("hog", "lbp", "hsv"), pca_components=pca, C=10.0, family_weighting="none")
    row, _, _ = fit_eval(spec, tr, va)
    row["experiment"] = "family_weighting_none"
    d["rows"].append(row)
    print(json.dumps({k: row[k] for k in ("pca", "top1", "top5", "macro_f1", "ece_15", "fit_s")}), flush=True)
res.write_text(json.dumps(d, indent=1, default=float))
