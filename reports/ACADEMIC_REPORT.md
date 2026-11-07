# Intelligent Fine-Grained Vehicle Recognition and Analysis System

*Technical and academic report — generated 2026-11-07 05:41 UTC from recorded artefacts by `scripts/generate_report.py`. Validation results come from a split of the official training set; official-test results are labelled **OFFICIAL TEST** and were computed once after configuration freeze.*

## Abstract

We build an end-to-end system that detects a vehicle in a photograph, lets the user select it, and classifies it into one of the 196 Stanford Cars make/model/year classes, with calibrated confidence and abstention. On the official test split, the deployed deep classifier reaches **94.12% top-1** and **98.94% top-5** accuracy on ground-truth vehicle crops; the classical HOG+LBP+HSV SVM baseline reaches 10.74% top-1. End-to-end details, calibration, robustness and failure cases follow.

## 1. Introduction and problem statement

Fine-grained vehicle recognition distinguishes classes that differ in small details (grille, lights, trim) and between model years of the same model. The task combines localisation (where is the car?) with fine-grained classification (which make, model and year?). The project also serves as a controlled comparison of classical computer vision, deep learning and their probability-level fusion, deployed on free CPU hardware.

## 2. Objectives and scope

Objectives: (i) a reproducible data pipeline on the original release; (ii) detector and classifier comparisons; (iii) a classical HOG/LBP/HSV + calibrated SVM branch; (iv) validated hybrid fusion; (v) calibrated confidence with rejection; (vi) robustness and failure analysis; (vii) a public CPU web application. Out of scope: number plates, tracking, traffic events, damage, valuation, ownership, surveillance.

## 3. Dataset description and integrity

Stanford Cars (Krause et al., 2013): 16,185 images, 196 classes, official split 8,144 train / 8,041 test (matches the official description). The original host was offline; the original files were retrieved from Internet Archive captures and are byte-identical to the official release (MD5 equal to torchvision's pinned values; `configs/dataset_sources.yaml`).

- Decode failures: 0; invalid/out-of-bounds boxes: 0/0
- Box convention verified empirically: 1-based inclusive (MATLAB) → canonical 0-based half-open
- Train images per class: min/max/mean [24, 68, 41.55]
- Duplicates: {'exact_sha256': 39, 'near_rejected': 13, 'near_confirmed': 7}; train↔test: {'exact_sha256': 19, 'near_rejected': 8, 'near_confirmed': 5}
- **Label conflicts:** 44 duplicate pairs carry different labels (e.g. Dodge Caliber Wagon 2007 vs 2012) — label noise in the source
- Working split: {'excluded': 24, 'test': 8041, 'train': 6905, 'val': 1215} (validation stratified, duplicate-grouped, from official train only)
Full report: `reports/dataset/dataset_validation_report.md`.

## 4. Related work

Hand-crafted descriptors: HOG (Dalal & Triggs, 2005) and LBP (Ojala et al., 2002), classified with SVMs (Cortes & Vapnik, 1995) and Platt-calibrated probabilities (Platt, 1999). Deep classifiers: MobileNetV3 (Howard et al., 2019) and EfficientNetV2 (Tan & Le, 2021). One-stage detection: YOLO (Redmon et al., 2016) and its Ultralytics successors YOLO11/YOLO26 (Jocher et al., Ultralytics). Calibration: temperature scaling (Guo et al., 2017). Selective classification: Geifman & El-Yaniv (2017). Published Stanford Cars results use different protocols (crops, resolution, pretraining, training length); we do not compare our numbers to them as directly equivalent.

## 5. System architecture

```text
upload → validation (signature, size, pixels, decode) → EXIF → RGB → YOLO detector (ONNX) → boxes →
user selection → ROI (+5% pad) → CNN (ONNX) → temperature → [optional α-fusion with HOG+LBP+HSV→SVM] →
reject policy → Top-5 + make/model/year
```
One detector and one classifier are loaded in production (`artifacts/deployment/manifest.json`, checksum-verified).

## 6. Classical computer vision methodology and results

ROI resized to 160×96; HOG (9 bins, 16px cells, 2×2 blocks, L2-Hys), uniform LBP (P=8, R=1, 4×4 grid), HSV joint histogram [8, 4, 4]. Standardisation and PCA fitted on training data only; RBF-SVM with Platt calibration (5-fold, training data). Validation results:

| experiment | features | PCA | weighting | dims | C | top-1 | top-5 | macro-F1 | ECE | SVM ms |
|---|---|---|---|---|---|---|---|---|---|---|
| C_selection | hog+lbp+hsv | — | equal | 1908 | 1.0 | 7.82% | 24.53% | 0.0393 | 0.0278 | 7.616 |
| C_selection | hog+lbp+hsv | — | equal | 1908 | 10.0 | 8.31% | 28.23% | 0.0407 | 0.0312 | 6.547 |
| C_selection | hog+lbp+hsv | — | equal | 1908 | 100.0 | 8.31% | 28.23% | 0.0407 | 0.0312 | 6.746 |
| family_ablation | hog | — | equal | 1620 | 10.0 | 9.55% | 28.40% | 0.0541 | 0.0414 | 6.01 |
| family_ablation | lbp | — | equal | 160 | 10.0 | 4.28% | 16.95% | 0.0177 | 0.0097 | 2.27 |
| family_ablation | hsv | — | equal | 128 | 10.0 | 3.05% | 11.44% | 0.0101 | 0.0031 | 2.131 |
| family_ablation | hog+lbp | — | equal | 1780 | 10.0 | 7.00% | 26.67% | 0.0374 | 0.0192 | 6.205 |
| family_ablation | hog+hsv | — | equal | 1748 | 10.0 | 8.64% | 26.09% | 0.0488 | 0.0370 | 5.848 |
| family_ablation | lbp+hsv | — | equal | 288 | 10.0 | 6.26% | 20.25% | 0.0257 | 0.0229 | 2.469 |
| pca_comparison | hog+lbp+hsv | 0.95 | equal | 476 | 10.0 | 8.07% | 28.89% | 0.0394 | 0.0287 | 2.908 |
| pca_comparison | hog+lbp+hsv | 256 | equal | 256 | 10.0 | 7.24% | 28.40% | 0.0348 | 0.0209 | 2.443 |
| preprocessing_clahe | hog+lbp+hsv | — | equal | 1908 | 10.0 | 8.31% | 28.31% | 0.0424 | 0.0310 | 6.322 |
| family_weighting_none | hog+lbp+hsv | — | none | 1908 | 10.0 | 8.97% | 30.78% | 0.0495 | 0.0320 | 6.445 |
| family_weighting_none | hog+lbp+hsv | 0.95 | none | 582 | 10.0 | 9.38% | 31.77% | 0.0515 | 0.0356 | 3.337 |

Selected: `{'families': 'hog+lbp+hsv', 'pca': 0.95, 'C': 10.0, 'family_weighting': 'none', 'clahe': False, 'rule': 'max validation top-1 among full-feature variants (C, PCA, family weighting); ties -> smaller model'}`. Observation: equal per-family weighting let weak LBP/HSV dilute HOG; plain standardisation (HOG dominates by dimensionality) performed better. PCA explained variance is not feature importance; importance is assessed by the family ablation.

## 7. Deep-learning methodology

ImageNet-pretrained torchvision backbones, single 196-way head; 2 head-only epochs then full fine-tuning with AdamW (backbone LR = 0.3 × head LR), cosine schedule with warm-up, label smoothing 0.1, AMP, early stopping on validation top-1. ROI policy: GT box + 5% padding, squashed to S×S, ImageNet normalisation. Moderate augmentation (flip, ±8° rotation, perspective, colour jitter, blur, random erasing, ±8% box jitter). Resumable checkpoints; data order is a function of (seed, epoch).

| run | arch | px | top-1 | top-5 | macro-F1 | ECE raw | ECE cal. | T | best ep | CPU ms |
|---|---|---|---|---|---|---|---|---|---|---|
| cls_effv2s_224 | efficientnet_v2_s | 224 | 92.84% | 98.85% | 0.9265 | 0.0949 | 0.0315 | 0.7775 | 29 | 47.9 |
| cls_effv2s_288 | efficientnet_v2_s | 288 | 94.32% | 98.77% | 0.9416 | 0.0990 | 0.0284 | 0.7715 | 35 | 74.2 |
| cls_mnv3l_224 | mobilenet_v3_large | 224 | 89.38% | 97.53% | 0.8902 | 0.3132 | 0.0409 | 0.5218 | 39 | 9.6 |
| cls_mnv3l_224_auglight | mobilenet_v3_large | 224 | 87.41% | 97.04% | 0.8715 | 0.3447 | 0.0509 | 0.5036 | 34 | nan |
| cls_mnv3l_224_cwloss | mobilenet_v3_large | 224 | 89.46% | 97.45% | 0.8908 | 0.3145 | 0.0469 | 0.5233 | 38 | nan |
| cls_mnv3l_224_s1 | mobilenet_v3_large | 224 | 89.14% | 97.61% | 0.8884 | 0.3069 | 0.0394 | 0.5296 | 29 | nan |
| cls_mnv3l_224_s2 | mobilenet_v3_large | 224 | 89.63% | 97.61% | 0.8929 | 0.3183 | 0.0414 | 0.5251 | 31 | nan |
| cls_mnv3l_288 | mobilenet_v3_large | 288 | 90.70% | 97.86% | 0.9032 | 0.3005 | 0.0340 | 0.5142 | 28 | 13.8 |

Seed variance (baseline, n=3): top-1 mean 89.38% ± 0.25%. Selected: `cls_effv2s_288` — highest validation top-1 = 0.9432; runs within 2x the baseline seed std (0.0025) treated as equivalent; among those the lowest cpu_ms_p50 wins

## 8. Detector methodology and results

YOLO26n and YOLO11n fine-tuned as single-class detectors (40 epochs, 640 px, batch 32, identical settings).

| model | mAP50 | mAP50-95 | P | R | miss rate | IoU median | CPU ms p50 | ONNX MB |
|---|---|---|---|---|---|---|---|---|
| det_yolo26n | 0.9948 | 0.9387 | 0.9984 | 0.9991 | 0.0008 | 0.9691 | 53.0 | 9.9 |
| det_yolo11n | 0.9949 | 0.9396 | 0.9984 | 0.9991 | 0.0008 | 0.9661 | 65.6 | 10.7 |

Selected `det_yolo26n`: miss rates within 0.5 pp of the best (0.0008) treated as equivalent: ['det_yolo26n', 'det_yolo11n']; lowest CPU p50 latency among them wins. mAP/P/R from Ultralytics val (conf=0.001); miss rate, IoU and CPU latency from our ONNX pipeline at the production threshold (0.25) on this Mac (Apple M1, ONNX Runtime CPU)

## 9. Hybrid fusion

P_hybrid = α·P_deep + (1−α)·P_classical, α selected on validation (NLL); reported numbers include a cross-fitted estimate (α fitted on one half of validation, scored on the other).

| method | top-1 | top-5 | macro-F1 | ECE | NLL | Brier |
|---|---|---|---|---|---|---|
| deep only (calibrated) | 94.32% | 98.77% | 0.9416 | 0.0183 | 0.3032 | 0.0964 |
| classical only (calibrated) | 9.38% | 31.77% | 0.0515 | 0.0204 | 4.2469 | 0.9655 |
| hybrid alpha=1.0 (alpha fit on same val; optimistic) | 94.32% | 98.77% | 0.9416 | 0.0183 | 0.3032 | 0.0964 |
| deep only, half-split eval | 94.31% | 98.76% | 0.9400 | 0.0241 | 0.3044 | 0.0968 |
| hybrid, cross-fitted alpha (mean alpha=1.00) | 94.31% | 98.76% | 0.9400 | 0.0241 | 0.3044 | 0.0968 |

**Production decision:** deep_only — cross-fitted hybrid vs deep: top-1 +0.0000, NLL -0.0000; classes improved/worsened 0/0. Rule: adopt hybrid only if cross-fitted top-1 gain >= 0.5 pp without NLL regression (classical branch adds CPU latency).

## 10. Training setup and reproducibility

Free Kaggle GPU (Tesla T4); environment recorded per run (`env.json`). Dependencies pinned (`uv.lock`, `requirements/train-cloud.txt`). Dataset identity pinned by SHA-256. Split seed and manifest hash recorded. Single training run per configuration except the baseline, which has three seeds.

## 11. Evaluation protocol

Model choice, temperature, fusion weight and rejection threshold were selected on validation only and frozen before a single evaluation on the official test split. ECE: top-label, 15 equal-width bins. Detection match: IoU ≥ 0.5. End-to-end success: highest-confidence box with IoU ≥ 0.5 and correct top-1 (denominator: all images).

## 12. Results on the official test split

Frozen commit `7dc0e0ae1661d6c3ed31c7a3ced2fe1e8dd23a61`, bundle `vr-yolo26n-efficientnet_v2_s-288-fp32-21bcac1`.

*Note: the repository history was squashed into a single commit on 2026-11-07, so the frozen commit is no longer on GitHub (it is kept in an archived backup). The evaluated code and configuration are byte-identical to the frozen state; see the appendix and `reports/results/history_provenance.json`.*

| method | n | top-1 | top-5 | macro-F1 | ECE | NLL |
|---|---|---|---|---|---|---|
| deep only (efficientnet_v2_s 288px, T=0.771) | 8041 | 94.12% | 98.94% | 0.9402 | 0.0159 | 0.2890 |
| classical HOG+LBP+HSV -> SVM | 8041 | 10.74% | 32.98% | 0.0647 | 0.0332 | 4.2054 |
| hybrid alpha=1.0 | 8041 | 94.12% | 98.94% | 0.9402 | 0.0159 | 0.2890 |
| deep only, excluding test images that duplicate train | 8017 | 94.26% | 98.95% | 0.9412 | 0.0145 | 0.2798 |

| track | value | denominator |
|---|---|---|
| A: classifier on GT ROI, top-1 | 94.12% | 8041 |
| B: classifier on matched detector ROI, top-1 | 94.14% | 8034 |
| B: same, misses counted as errors | 94.06% | 8041 |
| B: predictions changed by crop (GT vs detector box) | 1.13% | 8034 |
| C: end-to-end success | 94.02% | 8041 |


Detector on test: miss rate 0.09%, median IoU 0.9699, images with unmatched detections 0.44% (may be real but unannotated background cars).


Selective prediction with the frozen threshold 0.3970: coverage 98.88%, selective accuracy 94.78% (accuracy before rejection 94.12%).

## 13. Ablation studies

Feature-family, PCA and weighting ablations: §6. Architecture × resolution, augmentation strength and class-weighted loss: §7 table (runs `*_auglight`, `*_cwloss`). Deep vs hybrid: §9. FP32 vs INT8: §16.

## 14. Robustness evaluation

Synthetic perturbations on validation ROIs (not natural conditions; never used for tuning).

| condition | severity | top-1 | top-5 | ECE | mean conf. | classical top-1 | hybrid top-1 | detector hit |
|---|---|---|---|---|---|---|---|---|
| clean | 0 | 94.39% | 98.47% | 0.0221 | 0.9491 | 9.44% | 94.39% | 100.00% |
| lighting | 1 | 94.64% | 98.72% | 0.0234 | 0.9464 | 8.67% | 94.64% | 100.00% |
| lighting | 2 | 93.88% | 98.47% | 0.0282 | 0.9486 | 9.69% | 93.88% | 100.00% |
| lighting | 3 | 92.60% | 98.47% | 0.0335 | 0.9463 | 8.16% | 92.60% | 100.00% |
| occlusion | 1 | 93.62% | 98.47% | 0.0255 | 0.9354 | 7.14% | 93.62% | — |
| occlusion | 2 | 91.33% | 98.21% | 0.0440 | 0.9163 | 2.81% | 91.33% | — |
| occlusion | 3 | 85.71% | 97.45% | 0.0439 | 0.8865 | 1.02% | 85.71% | — |
| viewpoint | 1 | 94.64% | 98.72% | 0.0166 | 0.9476 | 8.16% | 94.64% | — |
| viewpoint | 2 | 93.62% | 98.72% | 0.0177 | 0.9436 | 5.87% | 93.62% | — |
| viewpoint | 3 | 92.09% | 97.96% | 0.0366 | 0.9238 | 1.79% | 92.09% | — |
| low_resolution | 1 | 93.62% | 98.98% | 0.0275 | 0.9449 | 8.42% | 93.62% | 100.00% |
| low_resolution | 2 | 83.42% | 94.39% | 0.0503 | 0.8629 | 6.63% | 83.42% | 99.74% |
| low_resolution | 3 | 48.21% | 67.09% | 0.1935 | 0.6756 | 3.83% | 48.21% | 98.47% |
| blur | 1 | 93.88% | 98.47% | 0.0235 | 0.9400 | 9.44% | 93.88% | 100.00% |
| blur | 2 | 92.09% | 98.21% | 0.0256 | 0.9297 | 8.42% | 92.09% | 100.00% |
| blur | 3 | 85.97% | 95.15% | 0.0290 | 0.8709 | 6.63% | 85.97% | 99.74% |
| noise_compression | 1 | 93.11% | 98.72% | 0.0280 | 0.9450 | 7.91% | 93.11% | 100.00% |
| noise_compression | 2 | 93.11% | 98.72% | 0.0331 | 0.9271 | 6.12% | 93.11% | 100.00% |
| noise_compression | 3 | 88.78% | 96.68% | 0.0401 | 0.8911 | 5.10% | 88.78% | 98.72% |

## 15. Confidence and calibration

| model | ECE | NLL | Brier | mean conf. |
|---|---|---|---|---|
| deep cls_effv2s_288 (raw softmax) | 0.0990 | 0.3823 | 0.1082 | 0.8447 |
| deep cls_effv2s_288 (T=0.771, cross-fitted) | 0.0284 | 0.3040 | 0.0974 | 0.9518 |
| classical SVM (Platt) | 0.0356 | 4.2601 | 0.9689 | 0.0582 |
| classical SVM (+T=0.860, cross-fitted) | 0.0270 | 4.2497 | 0.9657 | 0.0748 |

Rejection policy: largest coverage with validation selective accuracy >= 95% (deep-only calibrated probabilities). Validation: {'threshold': 0.39700397765922896, 'accuracy_before_rejection': 0.9432098765432099, 'coverage': 0.9901234567901235, 'selective_accuracy': 0.9501246882793017, 'rejection_rate': 0.00987654320987652}; cross-fitted estimate {'coverage': 0.9889546050591174, 'selective_accuracy': 0.9521584074686072, 'rejection_rate': 0.011045394940882636}.

## 16. Failure-case analysis

12 genuine cases (from 479 candidates, 473 wrong top-1). Category rules: use transparent, measurable rules (reported with each case) and are confirmed by visual review:
  similar_generation  wrong class with the same make+model text, different year/variant
  detector_miss       no detection with IoU >= 0.5 (or best IoU < 0.5)
  low_resolution      ground-truth box shorter side < 120 px
  lighting            ROI mean luminance < 60 or > 200
  viewpoint           box aspect ratio w/h < 1.15 (near-frontal / rear view)
  occlusion_truncation  GT box touches the image border (vehicle cut off) - truncation proxy for occlusion

| category | image | true | predicted | conf. | IoU | proposed improvement | tested |
|---|---|---|---|---|---|---|---|
| similar_generation | cars_test/06375.jpg | Dodge Ram Pickup 3500 Crew Cab 2010 | Dodge Ram Pickup 3500 Quad Cab 2009 | 0.9927 | 0.95 | higher input resolution / part-focused crops (grille, lights) or fine-grained attention; more examples of the confusable pair | False |
| similar_generation | cars_test/05760.jpg | Honda Accord Coupe 2012 | Honda Accord Sedan 2012 | 0.9926 | 0.969 | higher input resolution / part-focused crops (grille, lights) or fine-grained attention; more examples of the confusable pair | False |
| detector_miss | cars_test/07246.jpg | Rolls-Royce Phantom Drophead Coupe Convertible 2012 | Rolls-Royce Phantom Drophead Coupe Convertible 2012 | 0.9941 | 0.294 | lower detector threshold with user confirmation, or train with more small/occluded vehicles | False |
| detector_miss | cars_test/02277.jpg | Nissan NV Passenger Van 2012 | Nissan NV Passenger Van 2012 | 0.9885 | 0.231 | lower detector threshold with user confirmation, or train with more small/occluded vehicles | False |
| low_resolution | cars_test/05063.jpg | Dodge Caliber Wagon 2012 | Dodge Caliber Wagon 2007 | 0.9910 | 0.955 | super-resolution or training-time downscale augmentation; show low-resolution warning | False |
| low_resolution | cars_test/01161.jpg | Dodge Charger SRT-8 2009 | Dodge Charger Sedan 2012 | 0.9903 | 0.967 | super-resolution or training-time downscale augmentation; show low-resolution warning | False |
| lighting | cars_test/07940.jpg | Aston Martin V8 Vantage Coupe 2012 | Aston Martin Virage Coupe 2012 | 0.9784 | 0.984 | stronger photometric augmentation; exposure normalisation | False |
| lighting | cars_test/03159.jpg | Suzuki SX4 Sedan 2012 | Acura TSX Sedan 2012 | 0.8771 | 0.98 | stronger photometric augmentation; exposure normalisation | False |
| viewpoint | cars_test/06597.jpg | Bugatti Veyron 16.4 Coupe 2009 | Bugatti Veyron 16.4 Convertible 2009 | 0.9919 | 0.971 | more frontal/rear training views; viewpoint-balanced sampling | False |
| viewpoint | cars_test/02796.jpg | Spyker C8 Convertible 2009 | Spyker C8 Coupe 2009 | 0.9863 | 0.956 | more frontal/rear training views; viewpoint-balanced sampling | False |
| occlusion_truncation | cars_test/05757.jpg | Aston Martin V8 Vantage Coupe 2012 | Aston Martin Virage Coupe 2012 | 0.9916 | 0.935 | random-erasing/cutout at higher rates; occlusion-aware training | False |
| occlusion_truncation | cars_test/07470.jpg | GMC Savana Van 2012 | Chevrolet Express Van 2007 | 0.9890 | 0.974 | random-erasing/cutout at higher rates; occlusion-aware training | False |

## 17. Deployment and optimisation

CPU benchmark (macOS-26.6.2-arm64-arm-64bit, 2026-11-06T18:01:43Z): fresh process per row; 1 warm-up call (first_call_ms) + 3 discarded; P50/P95 over 40 calls; batch 1; peak RSS of the whole process

| kind | model | input | threads | MB | load ms | p50 ms | p95 ms | peak RSS MB |
|---|---|---|---|---|---|---|---|---|
| classifier | cls_mnv3l_224.onnx | 224 | 1 | 17.8 | 1435 | 14.6 | 15.4 | 155 |
| classifier | cls_mnv3l_224.onnx | 224 | 2 | 17.8 | 264 | 9.6 | 10.4 | 155 |
| classifier | cls_mnv3l_224.onnx | 224 | 4 | 17.8 | 273 | 7.2 | 8.2 | 162 |
| classifier | cls_mnv3l_288.onnx | 288 | 1 | 17.8 | 285 | 21.6 | 22.8 | 162 |
| classifier | cls_mnv3l_288.onnx | 288 | 2 | 17.8 | 266 | 13.8 | 14.1 | 167 |
| classifier | cls_mnv3l_288.onnx | 288 | 4 | 17.8 | 263 | 10.1 | 11.2 | 166 |
| classifier | cls_effv2s_224.onnx | 224 | 1 | 81.6 | 300 | 84.3 | 84.8 | 230 |
| classifier | cls_effv2s_224.onnx | 224 | 2 | 81.6 | 299 | 47.9 | 48.3 | 277 |
| classifier | cls_effv2s_224.onnx | 224 | 4 | 81.6 | 288 | 29.6 | 30.6 | 210 |
| classifier | cls_effv2s_288.onnx | 288 | 1 | 81.6 | 308 | 133.4 | 137.0 | 229 |
| classifier | cls_effv2s_288.onnx | 288 | 2 | 81.6 | 296 | 74.1 | 74.6 | 231 |
| classifier | cls_effv2s_288.onnx | 288 | 4 | 81.6 | 298 | 44.2 | 45.1 | 298 |
| detector | best.onnx | 640 | 1 | 9.9 | 154 | 87.4 | 88.0 | 209 |
| detector | best.onnx | 640 | 2 | 9.9 | 164 | 48.3 | 52.5 | 215 |
| detector | best.onnx | 640 | 4 | 9.9 | 152 | 29.5 | 30.6 | 215 |
| detector | best.onnx | 640 | 1 | 10.7 | 149 | 100.9 | 101.5 | 208 |
| detector | best.onnx | 640 | 2 | 10.7 | 147 | 56.4 | 56.7 | 211 |
| detector | best.onnx | 640 | 4 | 10.7 | 149 | 35.3 | 36.7 | 217 |

**NOT RUN** — hosted cold/warm latency measurement: no recorded artefact.

## 18. Limitations and threats to validity

- Single run per configuration except the seed-repeated baseline; small differences may not be significant.
- Validation (1,215 images, ~6 per class) is small; selection noise is non-trivial.
- Stanford Cars contains label conflicts between near-identical classes; accuracy has an irreducible ceiling.
- Detector trained/evaluated on one annotated car per image; multi-vehicle performance is not benchmarked.
- Robustness uses synthetic perturbations, not natural condition labels.
- Classes cover mostly 1991–2012 models; newer vehicles are out of distribution.
- Free hosting: cold starts and shared CPU; latency on the host differs from local measurements.

## 19. Conclusion and future work

See abstract and results. Future work: part-based or attention-based fine-grained models, higher resolution, external real-world evaluation set, and detector training on multi-vehicle scenes.

## References

- N. Dalal, B. Triggs. Histograms of Oriented Gradients for Human Detection. CVPR 2005.
- T. Ojala, M. Pietikäinen, T. Mäenpää. Multiresolution gray-scale and rotation invariant texture classification with local binary patterns. IEEE TPAMI 24(7), 2002.
- C. Cortes, V. Vapnik. Support-vector networks. Machine Learning 20, 1995.
- J. Platt. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. 1999.
- T.-F. Wu, C.-J. Lin, R. C. Weng. Probability estimates for multi-class classification by pairwise coupling. JMLR 5, 2004.
- J. Krause, M. Stark, J. Deng, L. Fei-Fei. 3D Object Representations for Fine-Grained Categorization. 3dRR-13 (ICCV workshops), 2013.
- J. Redmon, S. Divvala, R. Girshick, A. Farhadi. You Only Look Once: Unified, Real-Time Object Detection. CVPR 2016.
- G. Jocher et al. Ultralytics YOLO (YOLO11, YOLO26). Software, https://github.com/ultralytics/ultralytics.
- A. Howard et al. Searching for MobileNetV3. ICCV 2019.
- M. Tan, Q. Le. EfficientNetV2: Smaller Models and Faster Training. ICML 2021.
- C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger. On Calibration of Modern Neural Networks. ICML 2017.
- Y. Geifman, R. El-Yaniv. Selective Classification for Deep Neural Networks. NeurIPS 2017.

## Appendix: configuration and provenance

Freeze record: `"7dc0e0ae1661d6c3ed31c7a3ced2fe1e8dd23a61"`. Configs: `configs/`. Acceptance ledger: `milestone_acceptance.yaml`.

**History squash (2026-11-07).** On 2026-11-07 the repository history was squashed into a single root commit (8ca3b33, 'Initial commit'). The frozen commit recorded in freeze_record.json and final_test.json (7dc0e0a) is therefore no longer on GitHub; it is preserved in the author's archived pre-squash backup. Git object hashes are content hashes, so the entries below let anyone confirm, from the public repository alone, that the evaluated code and configuration are byte-identical to the frozen state.

Frozen tree: `0115c044b67f08711004fb5c616f9dd25a80b94f`. For each entry run `git rev-parse 8ca3b33:<path>` and compare with the hash recorded here.

| path | git object hash (frozen = current) |
|---|---|
| src | `ffe9063b4f4510fd622671caf252a848cb623fcb` |
| configs | `43802f041d7a239594b88f06536d54d5c070ea51` |
| tests | `f5c9526b701887c25d77905b281758dbc160d63f` |
| app.py | `b724fc49e2b586f286a921db824fcaddb59276bf` |
| pyproject.toml | `ae5d0eaee686eee0821b66024d074290635fd7e9` |
| uv.lock | `94838f29bd88b3dec9d805ee8e788740443383b0` |
| requirements.txt | `39a30ffc7008943737e1b070cf13e8d782d8a87f` |
| requirements | `faa5fea2ba3edc2d691d77f1d22afea3c8d8bd1c` |
| Dockerfile | `77c4d638248772ff0d2160274b6e42b8c6f3a169` |
| .streamlit | `d8de32d166d8dfcac5f509b27886f59bae2dbd0e` |

All 23 scripts present at the freeze are unchanged (per-file hashes in `history_provenance.json`). Only post-evaluation outputs (results, figures, report, ledger, README) and one plotting script (scripts/generate_figures.py, which reads recorded results and runs no model) were added or modified after the freeze.

