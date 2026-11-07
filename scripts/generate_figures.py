"""Report figures (PNG, 200 dpi) generated only from recorded artefacts in reports/results and the training
histories. Output: reports/figures/. Missing artefacts are skipped, never faked.

Palette: validated categorical order (blue, orange, aqua, yellow, magenta, green) - colour-blind checked; the lighter
slots are always paired with direct labels and distinct marker shapes (relief rule). One y-axis per chart.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "reports" / "results"
OUT = ROOT / "reports" / "figures"
RUNS = ROOT / "artifacts" / "kaggle_run1" / "runs"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARK = ["o", "s", "^", "D", "v", "P"]
INK, INK2, GRID, MUTED = "#0b0b0b", "#52514e", "#e6e5e1", "#9b9a95"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.titleweight": "bold",
    "axes.titlesize": 11, "axes.titlecolor": INK, "legend.frameon": False, "lines.linewidth": 2,
})


def load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def save(fig, name, note):
    fig.text(0.01, 0.005, note, fontsize=7.5, color=INK2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    print("wrote", name)


def classifier_bars():
    c = load("classifier_comparison.json")
    if not c:
        return
    df = pd.DataFrame(c["runs"]).sort_values("top1")
    fig, ax = plt.subplots(figsize=(7.5, 4))
    cols = [C[0] if r == c["selected"] else "#a9c6ea" for r in df.run]
    ax.barh(df.run, df.top1 * 100, color=cols, height=0.6)
    for y, v in enumerate(df.top1 * 100):
        ax.text(v + 0.3, y, f"{v:.1f}%", va="center", color=INK, fontsize=9)
    ax.set_xlim(80, 97)
    ax.set_xlabel("Top-1 accuracy (%)")
    ax.set_title("Classifier candidates - validation top-1 (selected in dark blue)")
    ax.grid(axis="y", visible=False)
    save(fig, "fig_classifier_comparison.png", "Split: validation (1,215 images from the official train split). Axis starts at 80%.")


def training_curves():
    runs = ["cls_effv2s_288", "cls_effv2s_224", "cls_mnv3l_288", "cls_mnv3l_224"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    any_ = False
    for i, r in enumerate(runs):
        p = RUNS / "classifiers" / r / "history.csv"
        if not p.exists():
            continue
        h = pd.read_csv(p)
        ax.plot(h.epoch, h.val_top1 * 100, color=C[i], marker=MARK[i], markevery=5, ms=5, label=r)
        any_ = True
    if not any_:
        plt.close(fig)
        return
    ax.set_xlim(0, 40)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation top-1 (%)")
    ax.set_title("Training progress of the four main classifier runs")
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "fig_training_curves.png", "Split: validation. First 2 epochs train the head only (backbone frozen).")


def reliability():
    cal = load("calibration.json")
    if not cal:
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1, label="perfect calibration")
    keys = [k for k in cal["reliability"] if k.startswith("deep")]
    for i, k in enumerate(keys):
        b = [x for x in cal["reliability"][k] if x["count"]]
        ax.plot([x["confidence"] for x in b], [x["accuracy"] for x in b], color=C[i], marker=MARK[i], ms=6, label=k)
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_title("Reliability diagram - selected classifier")
    ax.legend(loc="upper left", fontsize=8)
    save(fig, "fig_reliability.png", "Split: validation. 15 equal-width bins; empty bins omitted.")


def coverage():
    s = load("selective.json")
    if not s:
        return
    c = pd.DataFrame(s["curve"]).dropna()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(c.coverage * 100, c.selective_accuracy * 100, color=C[0])
    sm = s["summary"]
    ax.plot(sm["coverage"] * 100, sm["selective_accuracy"] * 100, marker="o", ms=8, color=C[1])
    ax.annotate(f"chosen threshold {sm['threshold']:.3f}\ncoverage {sm['coverage']:.1%}, accuracy {sm['selective_accuracy']:.1%}",
                (sm["coverage"] * 100, sm["selective_accuracy"] * 100), xytext=(-170, -40), textcoords="offset points",
                fontsize=8, color=INK, arrowprops={"arrowstyle": "-", "color": MUTED})
    ax.set_xlabel("Coverage: share of images answered (%)")
    ax.set_ylabel("Accuracy on answered images (%)")
    ax.set_title("Abstention trade-off (coverage vs selective accuracy)")
    save(fig, "fig_coverage_accuracy.png", "Split: validation. Threshold policy: largest coverage with selective accuracy >= 95%.")


def robustness():
    r = load("robustness.json")
    if not r:
        return
    df = pd.DataFrame(r["rows"])
    clean = df[df.condition == "clean"].top1.iloc[0] * 100
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    conds = [c for c in df.condition.unique() if c != "clean"]
    for i, cnd in enumerate(conds):
        g = df[df.condition == cnd].sort_values("severity")
        xs, ys = [0, *g.severity], [clean, *(g.top1 * 100)]
        ax.plot(xs, ys, color=C[i], marker=MARK[i], ms=6, label=cnd.replace("_", " "))
    ax.set_xticks([0, 1, 2, 3], ["clean", "mild", "medium", "severe"])
    ax.set_xlim(-0.1, 3.1)
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title("Robustness to synthetic perturbations (selected classifier)")
    ax.legend(loc="lower left", fontsize=8)
    save(fig, "fig_robustness.png", "Split: validation, 392 images (2 per class). Synthetic perturbations, not natural conditions.")


def classical_ablation():
    a = load("classical_ablation.json")
    if not a:
        return
    df = pd.DataFrame(a["rows"])
    df = df[df.experiment.isin(["family_ablation", "C_selection", "family_weighting_none"])]
    df = df[~((df.experiment == "C_selection") & (df.C != 10.0))]
    df["label"] = df.apply(lambda r: r.families + (" (plain scaling" + (", PCA)" if r.pca == r.pca and r.pca else ")")
                                                   if r.experiment == "family_weighting_none" else ""), axis=1)
    df = df.sort_values("top5")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    y = range(len(df))
    ax.barh([i + 0.2 for i in y], df.top5 * 100, height=0.36, color=C[0], label="top-5")
    ax.barh([i - 0.2 for i in y], df.top1 * 100, height=0.36, color=C[1], label="top-1")
    ax.set_yticks(list(y), df.label, fontsize=8)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Classical HOG / LBP / HSV feature ablation (calibrated RBF-SVM)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", visible=False)
    save(fig, "fig_classical_ablation.png", "Split: validation. Chance level = 0.5% top-1. Equal family weighting unless marked 'plain scaling'.")


def final_test():
    f = load("final_test.json")
    if not f:
        return
    df = pd.DataFrame(f["rows"])
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.barh(df.method.str.slice(0, 55), df.top1 * 100, color=C[0], height=0.5)
    for yy, v in enumerate(df.top1 * 100):
        ax.text(v + 0.8, yy, f"{v:.1f}%", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Top-1 accuracy (%)")
    ax.set_title("OFFICIAL TEST (8,041 images) - evaluated once after freeze")
    ax.grid(axis="y", visible=False)
    save(fig, "fig_final_test.png", f"Frozen commit {f['frozen_commit'][:7]}. Ground-truth crops.")


if __name__ == "__main__":
    for fn in (classifier_bars, training_curves, reliability, coverage, robustness, classical_ablation, final_test):
        fn()
