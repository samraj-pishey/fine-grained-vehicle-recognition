"""Refit the selected classical configuration on the TRAIN subset and save the model, validation probabilities
(for fusion) and its metrics. Selection rule (validation only): best top-1 among full-feature (HOG+LBP+HSV)
variants across C, PCA and family-weighting experiments in reports/results/classical_ablation.json."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_classical import fit_eval  # noqa: E402

from vehicle_recognition.classical.features import FeatureConfig  # noqa: E402
from vehicle_recognition.classical.model import ClassicalSpec, save_model, stack_families  # noqa: E402
from vehicle_recognition.config import PROJECT_ROOT, artifact_root, get_data_paths, load_yaml  # noqa: E402
from vehicle_recognition.data.annotations import build_class_mapping, load_class_names  # noqa: E402
from vehicle_recognition.evaluation.metrics import confusion, per_class_accuracy, top_confusions  # noqa: E402

dp = get_data_paths()
res = PROJECT_ROOT / "reports" / "results" / "classical_ablation.json"
d = json.loads(res.read_text())
full = [r for r in d["rows"] if r["families"] == "hog+lbp+hsv" and r["experiment"] != "preprocessing_clahe"]
best = max(full, key=lambda r: (round(r["top1"], 4), -r["model_bytes"]))
weighting = "none" if best["experiment"] == "family_weighting_none" else "equal"
pca = best["pca"] if best["pca"] == best["pca"] else None  # NaN -> None
spec = ClassicalSpec(families=("hog", "lbp", "hsv"), pca_components=pca, C=best["C"], family_weighting=weighting)
tr = dict(np.load(dp.processed / "features" / "train_v1.npz"))
va = dict(np.load(dp.processed / "features" / "val_v1.npz"))
row, pipe, p_val = fit_eval(spec, tr, va)
cfg = load_yaml("classical.yaml")
art = artifact_root() / "classical"
art.mkdir(parents=True, exist_ok=True)
save_model(art / "classical_svm.joblib", pipe, spec, FeatureConfig.from_dict(cfg["feature_config"]).to_dict(),
           list(stack_families(tr, spec.families)[1]), {"val_top1": row["top1"], "roi_pad_frac": cfg["roi_pad_frac"]})
np.savez_compressed(dp.processed / "classical_val_probs.npz", probs=p_val, labels=va["y"], relative_path=va["relative_path"])
names = [c.name for c in build_class_mapping(load_class_names(dp.devkit / "cars_meta.mat"))]
pc = per_class_accuracy(p_val.argmax(1), va["y"], 196)
d["selected"] = {"families": "hog+lbp+hsv", "pca": pca, "C": best["C"], "family_weighting": weighting, "clahe": False,
                 "rule": "max validation top-1 among full-feature variants (C, PCA, family weighting); ties -> smaller model"}
d["selected_metrics"] = {k: row[k] for k in ("top1", "top5", "macro_f1", "mean_per_class_acc", "worst_class_acc",
                                              "ece_15", "nll", "brier", "svm_ms", "feat_ms", "model_bytes")}
d["selected_per_class_acc"] = {names[i]: (None if np.isnan(v) else round(float(v), 4)) for i, v in enumerate(pc)}
d["selected_top_confusions"] = top_confusions(confusion(p_val, va["y"]), names, 15)
res.write_text(json.dumps(d, indent=1, default=float))
print(d["selected"], {k: round(v, 4) for k, v in d["selected_metrics"].items()})
