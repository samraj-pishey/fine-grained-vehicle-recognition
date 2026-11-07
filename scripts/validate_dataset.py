"""Discover, parse and validate the Stanford Cars dataset; build canonical metadata and the split.

Outputs (under DATA_ROOT):
  manifests/dataset_inventory.json, manifests/dataset_manifest.csv
  metadata/canonical_annotations.parquet (+ .csv), metadata/class_mapping.json, metadata/split_manifest.csv
  reports/class_distribution.csv, reports/exclusions.csv, reports/duplicate_audit.csv,
  reports/bbox_audit/*.jpg, reports/dataset_validation_report.md
Small, image-free summaries are mirrored into <repo>/reports/dataset/ for version control.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from vehicle_recognition.config import PROJECT_ROOT, get_data_paths, load_yaml
from vehicle_recognition.data.annotations import (
    NUM_CLASSES,
    build_canonical_table,
    build_class_mapping,
    load_class_names,
)
from vehicle_recognition.data.splits import duplicate_groups, make_train_val_split
from vehicle_recognition.data.validation import (
    box_checks,
    coordinate_convention_evidence,
    discover_structure,
    inspect_images,
    near_duplicate_candidates,
    ssim_pair,
    write_json,
)


def archive_inventory(dp) -> list[dict]:
    man = pd.read_csv(dp.manifests / "acquisition_manifest.tsv", sep="\t")
    out = []
    for _, r in man.iterrows():
        p = dp.raw_archives / r["file"]
        entry = {**r.to_dict(), "present": p.exists()}
        if p.suffix == ".tgz" and p.exists():
            with tarfile.open(p, "r:gz") as tf:
                members = tf.getmembers()
            entry["members"] = len(members)
            entry["member_files"] = sum(m.isfile() for m in members)
            entry["uncompressed_bytes"] = int(sum(m.size for m in members))
            entry["top_level"] = sorted({m.name.split("/")[0] for m in members})
        out.append(entry)
    return out


def draw_box(path: Path, box, label: str, out: Path) -> None:
    with Image.open(path) as im:
        im = im.convert("RGB")  # stored orientation: annotations refer to stored pixels
        d = ImageDraw.Draw(im)
        x0, y0, x1, y1 = box
        lw = max(2, round(min(im.size) / 150))
        d.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(255, 40, 40), width=lw)
        d.text((x0 + 4, max(0, y0 - 14)), label, fill=(255, 255, 0))
        im.thumbnail((800, 800))
        im.save(out, quality=88)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--overlays", type=int, default=24)
    args = ap.parse_args()
    cfg = load_yaml("dataset.yaml")
    dp = get_data_paths(args.data_root)
    t0 = time.time()
    for d in (dp.metadata, dp.reports, dp.manifests):
        d.mkdir(parents=True, exist_ok=True)
    repo_reports = PROJECT_ROOT / "reports" / "dataset"
    repo_reports.mkdir(parents=True, exist_ok=True)

    # ---------- 1. structure discovery ----------
    print("[1] structure discovery")
    inventory = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "archives": archive_inventory(dp),
        "raw_extracted": discover_structure(dp.raw_extracted),
    }

    # ---------- 2. annotations + class mapping ----------
    print("[2] annotation parsing")
    names = load_class_names(dp.devkit / "cars_meta.mat")
    mapping = build_class_mapping(names)
    df = build_canonical_table(dp.devkit, dp.test_labels_mat)
    write_json(
        {
            "source": "Stanford Cars devkit cars_meta.mat (Krause et al., 3dRR-13, 2013)",
            "num_classes": NUM_CLASSES,
            "class_id_convention": "0-based class_id = devkit class - 1",
            "make_model_year_rule": "year = trailing 4-digit token; make = first token except multi-word makes "
            "{AM General, Aston Martin, Land Rover}; model = remaining text (incl. body style) verbatim",
            "classes": [c.to_dict() for c in mapping],
        },
        dp.metadata / "class_mapping.json",
    )

    # ---------- 3. image inspection ----------
    print(f"[3] decoding {len(df)} referenced images with {args.workers} workers")
    paths = [dp.raw_extracted / rp for rp in df["relative_path"]]
    facts = inspect_images(paths, workers=args.workers)
    facts["relative_path"] = df["relative_path"].values
    df = df.join(facts.drop(columns=["path", "relative_path"]))
    # Every image file on disk should be referenced exactly once
    on_disk = {
        f"{sub}/{p.name}"
        for sub in ("cars_train", "cars_test")
        for p in (dp.raw_extracted / sub).glob("*")
        if p.is_file()
    }
    referenced = set(df["relative_path"])
    unreferenced = sorted(on_disk - referenced)
    missing = sorted(referenced - on_disk)
    ann_per_image = df.groupby("relative_path").size()
    multi_ann = ann_per_image[ann_per_image > 1]

    # ---------- 4. box checks ----------
    print("[4] box checks / coordinate convention")
    df = box_checks(df)
    conv = {s: coordinate_convention_evidence(g) for s, g in df[df["decodes"]].groupby("split")}

    # ---------- 5. duplicates ----------
    print("[5] duplicate audit (SHA-256 exact, pHash->SSIM near)")
    dup_rows = []
    sha_groups = df[df["sha256"] != ""].groupby("sha256")
    exact_pairs = []
    for _sha, g in sha_groups:
        if len(g) > 1:
            idx = g.index.tolist()
            for i in range(len(idx)):
                for j in range(i + 1, len(idx)):
                    exact_pairs.append((idx[i], idx[j]))
                    dup_rows.append(("exact_sha256", idx[i], idx[j], 0, 1.0))
    vcfg = cfg["validation"]
    cands = near_duplicate_candidates(df.reset_index(drop=True), max_dist=vcfg["near_duplicate_phash_max_distance"])
    exact_set = set(exact_pairs)
    confirmed_near = []
    for a, b, d in cands:
        if (a, b) in exact_set:
            continue
        s = ssim_pair(str(paths[a]), str(paths[b]))
        kind = "near_confirmed" if s >= vcfg["near_duplicate_ssim_min"] else "near_rejected"
        dup_rows.append((kind, a, b, d, s))
        if kind == "near_confirmed":
            confirmed_near.append((a, b))
    dup = pd.DataFrame(dup_rows, columns=["kind", "idx_a", "idx_b", "phash_distance", "ssim"])
    for side in ("a", "b"):
        dup[f"path_{side}"] = df.loc[dup[f"idx_{side}"], "relative_path"].values
        dup[f"split_{side}"] = df.loc[dup[f"idx_{side}"], "split"].values
        dup[f"class_{side}"] = df.loc[dup[f"idx_{side}"], "class_id"].values
    dup["cross_split"] = dup["split_a"] != dup["split_b"]
    dup["same_class"] = dup["class_a"] == dup["class_b"]
    dup.drop(columns=["idx_a", "idx_b"]).to_csv(dp.reports / "duplicate_audit.csv", index=False)
    df["dup_group"] = duplicate_groups(len(df), exact_pairs + confirmed_near)

    # ---------- 6. exclusions (recorded, never deleted) ----------
    excl = []
    for _, r in df.iterrows():
        reasons = []
        if not r["exists"]:
            reasons.append("missing_file")
        elif not r["decodes"]:
            reasons.append(f"decode_error:{r['error']}")
        else:
            if not r["box_positive"]:
                reasons.append("non_positive_box")
            if not r["box_in_bounds"]:
                reasons.append("box_out_of_bounds")
        if reasons:
            action = "exclude_from_training_and_report" if r["split"] == "train" else "keep_in_test_report_separately"
            excl.append({"relative_path": r["relative_path"], "split": r["split"], "reason": ";".join(reasons), "action": action})
    # cross-split duplicates: test copy stays in test; train copy is excluded from training/validation
    for _, r in dup[dup["cross_split"] & dup["kind"].isin(["exact_sha256", "near_confirmed"])].iterrows():
        tr_path = r["path_a"] if r["split_a"] == "train" else r["path_b"]
        excl.append(
            {"relative_path": tr_path, "split": "train", "reason": f"cross_split_duplicate:{r['kind']}",
             "action": "exclude_from_training_to_protect_test"}
        )
    excl_df = pd.DataFrame(excl, columns=["relative_path", "split", "reason", "action"]).drop_duplicates()
    excl_df.to_csv(dp.reports / "exclusions.csv", index=False)
    # test images that duplicate a training image (any label): kept in the official test set, but listed so the
    # final report can add a sensitivity analysis without them (never used for tuning)
    conf = dup[dup["cross_split"] & dup["kind"].isin(["exact_sha256", "near_confirmed"])]
    amb = pd.DataFrame({"relative_path": np.where(conf["split_a"] == "test", conf["path_a"], conf["path_b"]),
                        "label_conflict": ~conf["same_class"].values, "kind": conf["kind"].values}).drop_duplicates("relative_path")
    amb.to_csv(dp.reports / "test_images_duplicating_train.csv", index=False)
    df["excluded"] = df["relative_path"].isin(set(excl_df.loc[excl_df["split"] == "train", "relative_path"]))

    # ---------- 7. split ----------
    print("[7] train/val split from official train only")
    tr = df[(df["split"] == "train") & ~df["excluded"]]
    assign = make_train_val_split(tr, cfg["split"]["val_fraction"], cfg["split"]["seed"])
    df["subset"] = df["split"]  # 'test' stays 'test'
    df.loc[assign.index, "subset"] = assign
    df.loc[(df["split"] == "train") & df["excluded"], "subset"] = "excluded"
    df[["relative_path", "split", "subset", "class_id", "dup_group"]].to_csv(dp.metadata / "split_manifest.csv", index=False)

    # ---------- 8. outputs ----------
    keep = [
        "split", "subset", "fname", "relative_path", "class_id", "class_name", "x_min", "y_min", "x_max", "y_max",
        "width", "height", "annotation_valid", "source_annotation_id", "source_index", "source_class",
        "orig_x1", "orig_y1", "orig_x2", "orig_y2", "box_area_frac", "sha256", "phash", "mode", "format",
        "exif_orientation", "bytes", "decodes", "dup_group", "excluded",
    ]
    canon = df[keep]
    try:
        canon.to_parquet(dp.metadata / "canonical_annotations.parquet", index=False)
    except Exception as e:  # noqa: BLE001
        print("parquet failed, CSV only:", e)
    canon.to_csv(dp.metadata / "canonical_annotations.csv", index=False)
    df[["relative_path", "split", "bytes", "sha256", "format", "mode", "width", "height", "exif_orientation", "decodes", "error"]].to_csv(
        dp.manifests / "dataset_manifest.csv", index=False
    )
    dist = (
        df[df["split"].isin(["train", "test"])]
        .pivot_table(index="class_id", columns="subset", values="fname", aggfunc="count", fill_value=0)
        .reset_index()
    )
    dist.insert(1, "class_name", dist["class_id"].map({c.class_id: c.name for c in mapping}))
    dist["official_train"] = df[df["split"] == "train"].groupby("class_id").size().reindex(dist["class_id"]).values
    dist.to_csv(dp.reports / "class_distribution.csv", index=False)

    # overlays: random + edge cases (smallest/largest box fraction, out-of-bounds if any)
    ov_dir = dp.reports / "bbox_audit"
    ov_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    ok = df[df["decodes"]]
    sel = list(rng.choice(ok.index, size=min(args.overlays, len(ok)), replace=False))
    sel += ok.nsmallest(4, "box_area_frac").index.tolist() + ok.nlargest(4, "box_area_frac").index.tolist()
    sel += df[df["decodes"] & ~df["box_in_bounds"]].index.tolist()[:6]
    for i in dict.fromkeys(sel):
        r = df.loc[i]
        tag = "oob_" if not r["box_in_bounds"] else ""
        draw_box(paths[i], (r.x_min, r.y_min, r.x_max, r.y_max), f"{r.class_id}:{r.class_name}",
                 ov_dir / f"{tag}{r.split}_{Path(r.fname).stem}.jpg")

    # ---------- 9. report ----------
    counts = df.groupby("split").size().to_dict()
    sub_counts = df.groupby("subset").size().to_dict()
    per_class_train = df[df["split"] == "train"].groupby("class_id").size()
    stats = {
        "counts": counts,
        "subset_counts": sub_counts,
        "expected_counts": cfg["expected_counts"],
        "counts_match_official": {k: int(counts.get(k, 0)) == v for k, v in cfg["expected_counts"].items()},
        "classes_in_train": int(df.loc[df["split"] == "train", "class_id"].nunique()),
        "classes_in_test": int(df.loc[df["split"] == "test", "class_id"].nunique()),
        "class_id_range": [int(df["class_id"].min()), int(df["class_id"].max())],
        "missing_referenced_files": missing,
        "unreferenced_image_files": unreferenced[:50],
        "n_unreferenced_image_files": len(unreferenced),
        "images_with_multiple_annotations": int(len(multi_ann)),
        "decode_failures": int((~df["decodes"]).sum()),
        "non_positive_boxes": int((~df["box_positive"]).sum()),
        "out_of_bounds_boxes": int((df["decodes"] & ~df["box_in_bounds"]).sum()),
        "out_of_bounds_by_split": df[df["decodes"] & ~df["box_in_bounds"]].groupby("split").size().to_dict(),
        "max_oob_overshoot_px": float(
            np.maximum(df["x_max"] - df["width"], df["y_max"] - df["height"]).clip(lower=0).max()
        ),
        "coordinate_convention_evidence": conv,
        "modes": df["mode"].value_counts().to_dict(),
        "formats": df["format"].value_counts().to_dict(),
        "exif_orientation_not_1": int((df["exif_orientation"] != 1).sum()),
        "width": df["width"].describe().round(1).to_dict(),
        "height": df["height"].describe().round(1).to_dict(),
        "min_side_below_threshold": int((df[["width", "height"]].min(axis=1) < vcfg["min_side_px"]).sum()),
        "max_side_above_threshold": int((df[["width", "height"]].max(axis=1) > vcfg["max_side_px"]).sum()),
        "box_area_frac": df["box_area_frac"].describe().round(4).to_dict(),
        "per_class_train_min_max_mean": [int(per_class_train.min()), int(per_class_train.max()), round(float(per_class_train.mean()), 2)],
        "rarest_train_classes": per_class_train.nsmallest(5).rename(lambda c: mapping[c].name).to_dict(),
        "duplicates": dup["kind"].value_counts().to_dict() if len(dup) else {},
        "cross_split_duplicates": dup[dup["cross_split"]]["kind"].value_counts().to_dict() if len(dup) else {},
        "label_conflicting_duplicate_pairs": int((dup["kind"].isin(["exact_sha256", "near_confirmed"]) & ~dup["same_class"]).sum()) if len(dup) else 0,
        "label_conflicting_cross_split_pairs": int((dup["kind"].isin(["exact_sha256", "near_confirmed"]) & ~dup["same_class"] & dup["cross_split"]).sum()) if len(dup) else 0,
        "exif_rotating_tags": df.loc[df["exif_orientation"].isin([2, 3, 4, 5, 6, 7, 8]), "relative_path"].tolist(),
        "exclusions": excl_df["reason"].str.split(":").str[0].value_counts().to_dict() if len(excl_df) else {},
        "runtime_s": round(time.time() - t0, 1),
    }
    import hashlib
    stats["split_manifest_sha256"] = hashlib.sha256((dp.metadata / "split_manifest.csv").read_bytes()).hexdigest()
    inventory["validation_summary"] = stats
    write_json(inventory, dp.manifests / "dataset_inventory.json")

    rep = render_report(stats, inventory, dp)
    (dp.reports / "dataset_validation_report.md").write_text(rep)
    # repo mirror of the duplicate audit WITHOUT per-image class labels (labels are not redistributed)
    dup.drop(columns=["idx_a", "idx_b", "class_a", "class_b"]).to_csv(repo_reports / "duplicate_audit.csv", index=False)
    for f in [dp.reports / "dataset_validation_report.md", dp.reports / "class_distribution.csv",
              dp.reports / "exclusions.csv", dp.metadata / "class_mapping.json",
              dp.manifests / "dataset_inventory.json", dp.manifests / "acquisition_manifest.tsv"]:
        shutil.copy2(f, repo_reports / f.name)
    print(json.dumps({k: stats[k] for k in ["counts", "subset_counts", "decode_failures", "out_of_bounds_boxes",
                                             "duplicates", "cross_split_duplicates", "exclusions", "runtime_s"]}, indent=1, default=str))
    return 0


def render_report(s: dict, inv: dict, dp) -> str:
    ev = s["coordinate_convention_evidence"]
    arch = "\n".join(
        f"| `{a['file']}` | {a['bytes']:,} | `{a['sha256']}` | {a.get('member_files', '-')} | {a['retrieved_utc']} |"
        for a in inv["archives"]
    )
    lines = [
        "# Stanford Cars — Dataset Validation Report",
        "",
        f"Generated: {inv['generated_utc']} by `scripts/validate_dataset.py`.",
        "",
        "## 1. Source and provenance",
        "",
        "The official host `ai.stanford.edu/~jkrause/cars/` was unreachable on 2026-11-06 (TCP connection refused;"
        " torchvision documents the original URL as broken). With the user's approval, the **original Stanford files**"
        " were retrieved from Internet Archive raw-byte captures (`id_` URLs, capture 2022-12-12) of the official URLs.",
        "",
        "| File | Bytes | SHA-256 | Files in archive | Retrieved (UTC) |",
        "|---|---:|---|---:|---|",
        arch,
        "",
        "Terms (official page, archived capture): *\"This dataset is for research purposes only, similar to the ImageNet"
        " license.\"* Images are **not** redistributed in this repository or the deployed application.",
        "Citation: J. Krause, M. Stark, J. Deng, L. Fei-Fei. *3D Object Representations for Fine-Grained Categorization.*"
        " 4th IEEE Workshop on 3D Representation and Recognition (3dRR-13), ICCV 2013.",
        "",
        "## 2. Counts and split reconciliation",
        "",
        f"- Annotated images: {s['counts']} (official: {s['expected_counts']}) → match: {s['counts_match_official']}",
        f"- Classes present: train {s['classes_in_train']}, test {s['classes_in_test']}; class_id range {s['class_id_range']}",
        f"- Missing referenced files: {len(s['missing_referenced_files'])}; unreferenced image files: {s['n_unreferenced_image_files']}",
        f"- Images with more than one annotation: {s['images_with_multiple_annotations']} (each image has exactly one vehicle box in this release if 0)",
        f"- Working subsets: {s['subset_counts']} (validation drawn only from official train; seed in configs/dataset.yaml)",
        f"- Split manifest SHA-256 (`DATA_ROOT/metadata/split_manifest.csv`, not committed): `{s['split_manifest_sha256']}`",
        "",
        "## 3. Image integrity",
        "",
        f"- Decode failures (verify + full decode): {s['decode_failures']}",
        f"- Formats: {s['formats']}; colour modes: {s['modes']}",
        f"- Images with EXIF orientation ≠ 1: {s['exif_orientation_not_1']}",
        f"- Width stats: {s['width']}",
        f"- Height stats: {s['height']}",
        f"- Min side < threshold: {s['min_side_below_threshold']}; max side > threshold: {s['max_side_above_threshold']}",
        "",
        "## 4. Bounding boxes and coordinate convention",
        "",
        "The devkit README documents `bbox_x1..y2` as min/max pixel values. Empirical evidence per split:",
        "",
        "```json",
        json.dumps(ev, indent=1),
        "```",
        "",
        "Interpretation: if `orig_x1_eq_0` is 0 while `orig_x2_eq_width` is non-zero, the values are 1-based inclusive"
        " (MATLAB) indices. Canonical boxes are therefore `[x1-1, y1-1, x2, y2]` (0-based, half-open). Boxes whose"
        " original max coordinate exceeds the image size are recorded below; they are **clipped at use time** and"
        " listed in `exclusions.csv` only if invalid after clipping.",
        "",
        f"- Non-positive boxes: {s['non_positive_boxes']}",
        f"- Out-of-bounds (before clipping): {s['out_of_bounds_boxes']} {s['out_of_bounds_by_split']}; max overshoot {s['max_oob_overshoot_px']} px",
        f"- Box area / image area: {s['box_area_frac']}",
        "- Visual audit overlays: `DATA_ROOT/reports/bbox_audit/` (random sample + smallest/largest boxes + out-of-bounds cases;"
        " not committed because they contain dataset images).",
        "",
        "## 5. Class distribution",
        "",
        f"- Official-train images per class: min/max/mean = {s['per_class_train_min_max_mean']}",
        f"- Rarest train classes: {s['rarest_train_classes']}",
        "- Full table: `class_distribution.csv`.",
        "",
        "## 6. Duplicate and leakage audit",
        "",
        "Exact duplicates: SHA-256 of file bytes. Near duplicates: 64-bit pHash candidates (banded search, Hamming"
        " distance ≤ configured maximum; complete for distance ≤ 3), confirmed by SSIM on 256×256 grayscale.",
        "",
        f"- All duplicate pairs by kind: {s['duplicates']}",
        f"- Cross-split (train↔test) pairs by kind: {s['cross_split_duplicates']}",
        "- Policy: the test copy is untouched; the train copy of any confirmed cross-split duplicate is excluded from"
        " training/validation (`exclusions.csv`). Within-train duplicate clusters are kept on one side of the train/val split.",
        "",
        "### Label conflicts (dataset finding)",
        "",
        f"- Duplicate pairs (exact or confirmed near) whose two copies carry **different class labels**: "
        f"{s['label_conflicting_duplicate_pairs']} (of which train↔test: {s['label_conflicting_cross_split_pairs']}).",
        "  They involve visually near-identical classes (e.g. Dodge Caliber Wagon 2007 vs 2012, Chevrolet Express Cargo Van vs"
        " Express Van, Audi 100 vs Audi V8 Sedan 1994): the same photograph appears under two classes in the official release.",
        "  This is label noise / ambiguity in the source data. Training copies of cross-split duplicates are excluded; the"
        " affected test images remain in the official test set (protocol) and are listed in"
        " `test_images_duplicating_train.csv` for a sensitivity analysis in the final report.",
        "",
        "### EXIF orientation",
        "",
        f"- Images with a rotating EXIF tag: {s['exif_rotating_tags']}. Visual inspection shows the *stored* pixels are upright and"
        " the box matches them, i.e. the tag is wrong. All dataset loading uses stored pixels; the YOLO dataset gets an EXIF-free"
        " copy because OpenCV applies EXIF tags. The other images with tag ≠ 1 carry the invalid value 0 (ignored by readers).",
        "",
        "## 7. Exclusions",
        "",
        f"- By reason: {s['exclusions']} — nothing is deleted; see `exclusions.csv`.",
        "",
        f"Runtime: {s['runtime_s']} s.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
