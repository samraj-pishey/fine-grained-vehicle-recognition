# Data — Stanford Cars (not included)

The dataset is **research-only** and is never committed, packaged into Docker images, or uploaded to the Space.

## 1. Choose a data directory

Set `DATA_ROOT`, using either of these:

```bash
cp .env.example .env        # then edit DATA_ROOT=/absolute/or/relative/path
# or
export DATA_ROOT=/path/to/stanford_cars
```

The default in `configs/dataset.yaml` is `data_root/` inside the repository, which is git-ignored. Layout:

```text
<DATA_ROOT>/
├── raw_archives/   original archives (read-only; never modified)
├── raw_extracted/  exact extraction (read-only): cars_train/ cars_test/ devkit/ test_labels/
├── manifests/      acquisition_manifest.tsv (URLs, UTC time, bytes, SHA-256), dataset_inventory.json, dataset_manifest.csv
├── metadata/       canonical_annotations.{parquet,csv}, class_mapping.json, split_manifest.csv
├── processed/      deterministic derived data (features, logits)
├── roi_cache/      regenerable crops
├── cloud_package/  train/val-only table for Colab/Kaggle (no test rows)
└── reports/        validation report, class distribution, duplicate audit, exclusions, bbox overlays
```

## 2. Acquire the original files

The official host (`ai.stanford.edu/~jkrause/cars/car_dataset.html`) has been offline since about 2023, and
torchvision documents the original URL as broken. `scripts/acquire_dataset.sh` downloads the **original
Stanford files** from Internet Archive raw-byte captures of the official URLs (capture `20221212052731`), with
resume and retries, and records provenance and SHA-256 hashes:

```bash
make data            # = scripts/acquire_dataset.sh $DATA_ROOT && scripts/extract_dataset.sh $DATA_ROOT
make validate-data   # = python scripts/validate_dataset.py
```

Expected SHA-256 values are pinned in `configs/dataset_sources.yaml`. If your hashes differ, stop and investigate
before using the data.

If the Internet Archive is slow, `scripts/parallel_range_fetch.sh` completes a partial download with parallel
HTTP range requests. The integrity checks are the same: `gzip -t`, the archive member counts, and SHA-256.

## 3. Do not keep the project in an iCloud-synced folder

On macOS, ~/Desktop and ~/Documents are often synced to iCloud with "Optimise Mac Storage". That setting silently
replaces files with cloud placeholders ("dataless" files), causing read timeouts and corrupt-looking images
mid-run, and it also uploads the research-only dataset to iCloud. Keep the repository and `DATA_ROOT` in a
non-synced folder such as `~/Projects/` (check with `ls -lO`: `dataless` in the flags column means evicted).

## 4. Disk budget (measured)

Raw archives ≈ 1.87 GB, extracted ≈ 1.9 GB, ROI cache ≈ 0.3–0.6 GB, features and logits < 0.3 GB.
`scripts/extract_dataset.sh` refuses to extract if doing so would leave less than 5 GB free.
`make clean-caches` lists the regenerable caches and asks for confirmation before deleting anything.
Raw data is never deleted.
