# Stanford Cars — Dataset Validation Report

Generated: 2026-11-07T03:03:18+00:00 by `scripts/validate_dataset.py`.

## 1. Source and provenance

The official host `ai.stanford.edu/~jkrause/cars/` was unreachable on 2026-11-06 (TCP connection refused; torchvision documents the original URL as broken). With the user's approval, the **original Stanford files** were retrieved from Internet Archive raw-byte captures (`id_` URLs, capture 2022-12-12) of the official URLs.

| File | Bytes | SHA-256 | Files in archive | Retrieved (UTC) |
|---|---:|---|---:|---|
| `car_devkit.tgz` | 330,960 | `b97deb463af7d58b6bfaa18b2a4de9829f0f79e8ce663dfa9261bf7810e9accd` | 6 | 2026-11-06T09:56:11Z |
| `cars_test_annos_withlabels.mat` | 185,758 | `790f75be8ea34eeded134cc559332baf23e30e91367e9ddca97d26ed9b895f05` | - | 2026-11-06T09:56:13Z |
| `cars_train.tgz` | 979,269,282 | `512b227b30e2f0a8aab9e09485786ab4479582073a144998da74d64b801fd288` | 8144 | 2026-11-06T10:38:01Z |
| `cars_test.tgz` | 977,350,468 | `bffea656d6f425cba3c91c6d83336e4c5f86c6cffd8975b0f375d3a10da8e243` | 8041 | 2026-11-06T10:38:07Z |

Terms (official page, archived capture): *"This dataset is for research purposes only, similar to the ImageNet license."* Images are **not** redistributed in this repository or the deployed application.
Citation: J. Krause, M. Stark, J. Deng, L. Fei-Fei. *3D Object Representations for Fine-Grained Categorization.* 4th IEEE Workshop on 3D Representation and Recognition (3dRR-13), ICCV 2013.

## 2. Counts and split reconciliation

- Annotated images: {'test': 8041, 'train': 8144} (official: {'train': 8144, 'test': 8041}) → match: {'train': True, 'test': True}
- Classes present: train 196, test 196; class_id range [0, 195]
- Missing referenced files: 0; unreferenced image files: 0
- Images with more than one annotation: 0 (each image has exactly one vehicle box in this release if 0)
- Working subsets: {'excluded': 24, 'test': 8041, 'train': 6905, 'val': 1215} (validation drawn only from official train; seed in configs/dataset.yaml)
- Split manifest SHA-256 (`DATA_ROOT/metadata/split_manifest.csv`, not committed): `1127fb326fdf37dab4adf206195baaedbee15a26f6005983f73dc9df98baf203`

## 3. Image integrity

- Decode failures (verify + full decode): 0
- Formats: {'JPEG': 16185}; colour modes: {'RGB': 16151, 'L': 34}
- Images with EXIF orientation ≠ 1: 33
- Width stats: {'count': 16185.0, 'mean': 700.5, 'std': 453.2, 'min': 78.0, '25%': 429.0, '50%': 640.0, '75%': 800.0, 'max': 7800.0}
- Height stats: {'count': 16185.0, 'mean': 483.2, 'std': 318.3, 'min': 41.0, '25%': 290.0, '50%': 426.0, '75%': 540.0, 'max': 5400.0}
- Min side < threshold: 0; max side > threshold: 1

## 4. Bounding boxes and coordinate convention

The devkit README documents `bbox_x1..y2` as min/max pixel values. Empirical evidence per split:

```json
{
 "test": {
  "n": 8041,
  "orig_x1_eq_0": 0,
  "orig_y1_eq_0": 0,
  "orig_x1_eq_1": 160,
  "orig_y1_eq_1": 101,
  "orig_x2_eq_width": 207,
  "orig_y2_eq_height": 177,
  "orig_x2_gt_width": 0,
  "orig_y2_gt_height": 0,
  "orig_x2_eq_width_minus_1": 127
 },
 "train": {
  "n": 8144,
  "orig_x1_eq_0": 0,
  "orig_y1_eq_0": 0,
  "orig_x1_eq_1": 179,
  "orig_y1_eq_1": 102,
  "orig_x2_eq_width": 224,
  "orig_y2_eq_height": 170,
  "orig_x2_gt_width": 0,
  "orig_y2_gt_height": 0,
  "orig_x2_eq_width_minus_1": 133
 }
}
```

Interpretation: if `orig_x1_eq_0` is 0 while `orig_x2_eq_width` is non-zero, the values are 1-based inclusive (MATLAB) indices. Canonical boxes are therefore `[x1-1, y1-1, x2, y2]` (0-based, half-open). Boxes whose original max coordinate exceeds the image size are recorded below; they are **clipped at use time** and listed in `exclusions.csv` only if invalid after clipping.

- Non-positive boxes: 0
- Out-of-bounds (before clipping): 0 {}; max overshoot 0.0 px
- Box area / image area: {'count': 16185.0, 'mean': 0.5504, 'std': 0.1702, 'min': 0.0318, '25%': 0.4303, '50%': 0.5469, '75%': 0.6685, 'max': 1.0}
- Visual audit overlays: `DATA_ROOT/reports/bbox_audit/` (random sample + smallest/largest boxes + out-of-bounds cases; not committed because they contain dataset images).

## 5. Class distribution

- Official-train images per class: min/max/mean = [24, 68, 41.55]
- Rarest train classes: {'Hyundai Accent Sedan 2012': 24, 'FIAT 500 Abarth 2012': 28, 'Maybach Landaulet Convertible 2012': 29, 'Chevrolet Express Cargo Van 2007': 30, 'Rolls-Royce Phantom Drophead Coupe Convertible 2012': 31}
- Full table: `class_distribution.csv`.

## 6. Duplicate and leakage audit

Exact duplicates: SHA-256 of file bytes. Near duplicates: 64-bit pHash candidates (banded search, Hamming distance ≤ configured maximum; complete for distance ≤ 3), confirmed by SSIM on 256×256 grayscale.

- All duplicate pairs by kind: {'exact_sha256': 39, 'near_rejected': 13, 'near_confirmed': 7}
- Cross-split (train↔test) pairs by kind: {'exact_sha256': 19, 'near_rejected': 8, 'near_confirmed': 5}
- Policy: the test copy is untouched; the train copy of any confirmed cross-split duplicate is excluded from training/validation (`exclusions.csv`). Within-train duplicate clusters are kept on one side of the train/val split.

### Label conflicts (dataset finding)

- Duplicate pairs (exact or confirmed near) whose two copies carry **different class labels**: 44 (of which train↔test: 23).
  They involve visually near-identical classes (e.g. Dodge Caliber Wagon 2007 vs 2012, Chevrolet Express Cargo Van vs Express Van, Audi 100 vs Audi V8 Sedan 1994): the same photograph appears under two classes in the official release.
  This is label noise / ambiguity in the source data. Training copies of cross-split duplicates are excluded; the affected test images remain in the official test set (protocol) and are listed in `test_images_duplicating_train.csv` for a sensitivity analysis in the final report.

### EXIF orientation

- Images with a rotating EXIF tag: ['cars_train/07389.jpg']. Visual inspection shows the *stored* pixels are upright and the box matches them, i.e. the tag is wrong. All dataset loading uses stored pixels; the YOLO dataset gets an EXIF-free copy because OpenCV applies EXIF tags. The other images with tag ≠ 1 carry the invalid value 0 (ignored by readers).

## 7. Exclusions

- By reason: {'cross_split_duplicate': 24} — nothing is deleted; see `exclusions.csv`.

Runtime: 42.8 s.
