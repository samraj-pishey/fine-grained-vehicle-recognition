<div align="center">

# Intelligent Fine-Grained Vehicle Recognition and Analysis System

**P5 · Team T15 — Pixel**

A computer-vision system for detecting vehicles and recognising fine-grained make, model and year categories across the **196 classes of the Stanford Cars dataset**.

[Live application](https://vehicle-recognition.streamlit.app) · [Source repository](https://github.com/samraj-pishey/fine-grained-vehicle-recognition) · [Model release](https://github.com/samraj-pishey/fine-grained-vehicle-recognition/releases/tag/models-v1)

</div>

---

## Project overview

Upload a vehicle photograph, detect one or more vehicles, select the vehicle of interest, and receive a predicted make/model/year class, calibrated confidence and Top-5 alternatives. The application can abstain when the maximum calibrated confidence is below a threshold selected using validation data.

The project evaluates three approaches:

| Approach | Method |
|---|---|
| **Classical computer vision** | HOG, uniform LBP and HSV features, followed by PCA and an RBF-SVM |
| **Deep learning** | MobileNetV3-Large and EfficientNetV2-S image classifiers |
| **Hybrid probability fusion** | Validation-evaluated fusion of classical and deep predictions |

> **Scope:** This is a closed-set research system. It recognises only the 196 Stanford Cars classes. It is not a general-purpose vehicle identification, number-plate recognition, tracking, valuation or ownership-identification system.

## At a glance

| Component | Production configuration |
|---|---|
| Dataset | Stanford Cars |
| Classes | 196 |
| Vehicle detector | YOLO26n, ONNX FP32 |
| Fine-grained classifier | EfficientNetV2-S, 288 × 288, ONNX FP32 |
| Inference mode | Deep-only |
| Calibration | Temperature scaling, `T = 0.771459` |
| Confidence rejection threshold | `0.397004` |
| Runtime | ONNX Runtime CPU |
| Web application | Streamlit Community Cloud |

The production pipeline is **YOLO26n FP32 ONNX + EfficientNetV2-S 288 FP32 ONNX**. The classical classifier remains an evaluated baseline. It does not contribute to production probability fusion: the selected fusion weight was `α = 1.0`, giving the classical branch zero weight.

## Contents

- [How the system works](#how-the-system-works)
- [Dataset and evaluation protocol](#dataset-and-evaluation-protocol)
- [Results](#results)
  - [Vehicle detector comparison](#1-vehicle-detector-comparison)
  - [Fine-grained classifier comparison](#2-fine-grained-classifier-comparison)
  - [Classical, deep-only and hybrid comparison](#3-classical-deep-only-and-hybrid-comparison)
  - [Official-test classification and pipeline results](#4-official-test-classification-and-pipeline-results)
  - [Calibration and confidence-based abstention](#5-calibration-and-confidence-based-abstention)
  - [Robustness evaluation](#6-robustness-evaluation)
  - [Runtime and optimisation](#7-runtime-and-optimisation)
  - [Failure analysis](#8-failure-analysis)
- [P5 requirement coverage and known gaps](#p5-requirement-coverage-and-known-gaps)
- [Repository layout](#repository-layout)
- [Run locally](#run-locally)
- [Docker](#docker)
- [Limitations](#limitations)
- [Dataset, citation and licence](#dataset-citation-and-licence)

---
<img width="3795" height="2585" alt="P5_T15_System_Architecture" src="https://github.com/user-attachments/assets/6e773eda-056b-4d09-ac28-8714c0fabc61" />


## How the system works

```text
                    IMAGE UPLOAD
                         |
                         v
                  Input validation
           JPEG/PNG signature, size and decode
                         |
                         v
              EXIF orientation correction
                   and RGB conversion
                         |
                         v
              YOLO26n vehicle detector
                 ONNX Runtime · CPU
                         |
                         v
                Numbered bounding boxes
                         |
                         v
                User selects a vehicle
                         |
                         v
                 ROI crop + 5% padding
                         |
                         v
             EfficientNetV2-S classifier
                    288 × 288
                         |
                         v
                Temperature scaling
                    T = 0.771459
                         |
                         v
              Confidence threshold check
                    0.397004
                    /       \
                   /         \
                  v           v
          Confidence passes   Confidence below
                  |           threshold
                  v           v
             Top-5 results    Abstain / low-confidence
          make-model-year     uncertainty indication
```

The application validates uploaded images before inference, corrects EXIF orientation, detects vehicles, and lets the user select the region of interest (ROI). The selected crop is classified and calibrated. If its maximum calibrated confidence is below the configured threshold, the application can return a low-confidence/abstention result instead of presenting the prediction as sufficiently reliable.

---

## Dataset and evaluation protocol

The project uses the **Stanford Cars** dataset, containing 16,185 images across 196 classes.

| Partition | Images | Role |
|---|---:|---|
| Official training partition | 8,144 | Source training data |
| Validation holdout | 1,215 | Held-out subset of official training data; model selection, calibration and threshold selection |
| Remaining training portion | 6,929 | Training after the validation holdout |
| Official test partition | 8,041 | Final evaluation after configuration freeze |
| **Total original dataset** | **16,185** | Official dataset total |

The 1,215 validation images are included in the 8,144-image official training partition; they are not an additional dataset partition. The official test partition is kept separate.

The dataset audit records:

- Counts matching the official release: 8,144 training images, 8,041 test images and 196 classes.
- 24 exact duplicate images removed from the training working split; the official test partition was preserved.
- 44 duplicate photo pairs with conflicting labels in the source data.

The dataset is not redistributed in this repository. See [`data/README.md`](data/README.md) for acquisition and verification instructions.

### Reporting conventions

- Validation results are used for model selection, calibration and threshold selection.
- Official-test results are reported separately and were computed after the configuration freeze.
- Detector validation mAP is not presented as official-test mAP.
- Standalone model timings and pipeline-stage timings are different measurement protocols.
- Missing measurements are labelled **not measured**, rather than estimated.
- Synthetic robustness tests are not presented as real-world field validation.
- Metrics with different sample populations or denominators are reported separately.

Detailed experiment records are stored under [`reports/results/`](reports/results/). The academic report contains additional methodology and official-test evaluation details: [`reports/ACADEMIC_REPORT.md`](reports/ACADEMIC_REPORT.md).

---

# Results

## 1. Vehicle detector comparison

Validation results on the 1,215-image validation holdout:

| Detector | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Median IoU | CPU P50 | CPU P95 | ONNX size |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **YOLO26n (selected)** | **99.48%** | 93.87% | 99.84% | 99.91% | 0.9691 | **53.02 ms** | **58.25 ms** | **9.852 MB** |
| YOLO11n | 99.49% | 93.96% | 99.84% | 99.91% | 0.9661 | 65.64 ms | 70.85 ms | 10.663 MB |

The detector metrics are close. YOLO11n has marginally higher validation mAP, while YOLO26n has lower measured local CPU latency, a smaller ONNX artifact and a slightly higher median IoU. YOLO26n was selected under the recorded detector-selection rule: miss rates within 0.5 percentage points of the best were treated as equivalent, after which local latency and artifact size were considered.

### YOLO26n official-test localisation

| Metric | Result |
|---|---:|
| Official test images | 8,041 |
| Median IoU | 0.96990 |
| Mean IoU | 0.95761 |
| Miss rate (IoU < 0.5) | 0.0871% |
| No-detection rate | 0.0249% |
| Official-test mAP@0.5 | **Not measured** |

The miss rate and no-detection rate are reported separately. They should not be combined into a derived success rate without confirming the underlying event definitions and denominators. The official-test IoU statistics do not constitute an official-test mAP measurement.

## 2. Fine-grained classifier comparison

Validation results using ground-truth vehicle regions of interest (ROIs):

| Classifier | Input | Top-1 | Top-5 | Macro-F1 | CPU P50 | CPU P95 | ONNX size |
|---|---:|---:|---:|---:|---:|---:|---:|
| **EfficientNetV2-S (selected)** | 288 px | **94.32%** | 98.77% | **0.9416** | 74.15 ms | 74.61 ms | 81.562 MB |
| EfficientNetV2-S | 224 px | 92.84% | 98.85% | 0.9265 | 47.92 ms | 48.35 ms | 81.600 MB |
| MobileNetV3-Large | 288 px | 90.70% | 97.86% | 0.9032 | 13.75 ms | 14.08 ms | 17.802 MB |
| MobileNetV3-Large | 224 px | 89.38% | 97.53% | 0.8902 | 9.64 ms | 10.44 ms | 17.802 MB |

Additional MobileNetV3-Large training variants are recorded in the experiment results. EfficientNetV2-S at 288 px was selected for its higher validation Top-1 accuracy and Macro-F1. MobileNetV3-Large is smaller and faster in the recorded standalone benchmark, but less accurate.

The classifier timings are standalone local Mac M1 ONNX CPU measurements. They are not hosted application response times.

## 3. Classical, deep-only and hybrid comparison

### Classical computer-vision pipeline

```text
HOG + uniform LBP + HSV
           |
           v
     Standardisation
           |
           v
    PCA (95% variance)
           |
           v
 RBF-SVM (C = 10)
 + Platt calibration
```

Validation comparison:

| Method | Top-1 | Top-5 | Macro-F1 |
|---|---:|---:|---:|
| Deep-only, calibrated | 94.32% | 98.77% | 0.9416 |
| Classical-only | 9.38% | 31.77% | 0.0515 |
| Hybrid, selected α = 1.0 | 94.32% | 98.77% | 0.9416 |

Hybrid fusion was evaluated using:

\[
P = \alpha P_{\text{deep}} + (1-\alpha)P_{\text{classical}}
\]

The fusion weight was selected by validation NLL search over `α = 0.00` to `1.00` in increments of `0.05`. The selected value and cross-fitted mean value were both `α = 1.00`.

**Finding:** the experiment showed no improvement over deep-only classification. The production system therefore uses deep-only inference. The classical branch is retained as a comparative baseline and analytical component.

The full classical feature ablation is available in [`reports/results/classical_ablation.json`](reports/results/classical_ablation.json).

## 4. Official-test classification and pipeline results

Official-test classifier results on ground-truth ROIs:

| Metric | Result |
|---|---:|
| Samples | 8,041 |
| Top-1 accuracy | 94.12% |
| Top-5 accuracy | 98.94% |
| Macro-F1 | 0.9402 |

The pipeline evaluation distinguishes between ground-truth ROIs, matched detector-produced ROIs, and end-to-end detection-plus-classification success. These measurements have different denominators and must not be conflated.

| Evaluation track | Samples | Top-1 / success | Top-5 | Macro-F1 |
|---|---:|---:|---:|---:|
| Classifier on ground-truth ROI | 8,041 | 94.12% | 98.94% | 0.9402 |
| Classifier on matched predicted ROI | 8,034 | 94.14% | 98.84% | 0.9404 |
| Predicted ROI, all images; detector misses counted as errors | 8,041 | 94.06% | — | — |
| End-to-end detection + correct Top-1 | 8,041 | 94.02% | — | — |

For the end-to-end success measure, the top-confidence predicted box must have IoU ≥ 0.5 with the ground-truth box and the predicted Top-1 class must be correct.

## 5. Calibration and confidence-based abstention

Temperature scaling uses the frozen production temperature:

`T = 0.771459`

Validation calibration for EfficientNetV2-S 288:

| Metric | Before calibration | After cross-fitted calibration |
|---|---:|---:|
| ECE (15 equal-width bins) | 0.0990 | 0.0284 |
| NLL | 0.3823 | 0.3040 |
| Brier score | 0.1082 | 0.0974 |
| Mean confidence | 0.8447 | 0.9518 |

The confidence rejection threshold is `0.397004`. It was selected using validation data as the largest-coverage threshold satisfying the validation selective-accuracy target of at least 95%.

| Split | Samples | Ordinary Top-1 | Selective accuracy | Coverage | Accepted | Rejected |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 1,215 | 94.32% | 95.01% | 99.01% | 1,203 | 12 |
| Official test | 8,041 | 94.12% | 94.78% | 98.88% | 7,951 | 90 |

> **Evaluation note:** The 95% selective-accuracy target was achieved on validation but not on the official test. The threshold is not a guarantee that every accepted prediction is correct.

Official-test calibration for the frozen deep model:

| Metric | Result |
|---|---:|
| ECE (15 bins) | 0.0159 |
| NLL | 0.2890 |
| Brier score | 0.0955 |
| Mean confidence | 0.9539 |

## 6. Robustness evaluation

Robustness was evaluated on 392 validation ROIs (two images per class across 196 classes). The same 392 images were reused across six synthetic perturbation families and three severity levels, resulting in **7,056 perturbed-image evaluations**. Including the 392 clean baseline evaluations, the experiment covered **7,448 image evaluations**.

| Condition | Severity 1 Top-1 | Severity 2 Top-1 | Severity 3 Top-1 |
|---|---:|---:|---:|
| Lighting | 94.64% | 93.88% | 92.60% |
| Occlusion | 93.62% | 91.33% | 85.71% |
| Viewpoint | 94.64% | 93.62% | 92.09% |
| Low resolution | 93.62% | 83.42% | 48.21% |
| Blur | 93.88% | 92.09% | 85.97% |
| Noise and compression | 93.11% | 93.11% | 88.78% |

Perturbations include synthetic lighting changes and shadows, rectangular occlusion, perspective warps, downsample/upsample degradation, blur, Gaussian noise and JPEG compression.

These are **synthetic validation experiments**, not measurements on naturally collected adverse-condition photographs or the official test set. The planned sample size was 600; the completed evaluation used 392 images.

## 7. Runtime and optimisation

### Standalone local Mac M1 ONNX CPU benchmark

Batch size 1, two intra-op threads, warm-up followed by 40 timed calls:

| Model | Input | P50 | P95 |
|---|---:|---:|---:|
| YOLO26n FP32 ONNX | 640 px | 48.28 ms | 52.48 ms |
| EfficientNetV2-S 288 FP32 ONNX | 288 px | 74.15 ms | 74.61 ms |

These are standalone model timings. Separate validation and official-test pipeline-stage measurements are recorded in the academic report. Detector and classifier percentiles must not be added to infer an end-to-end P50 or P95.

### Optimisation experiments

| Experiment | Recorded result | Production status |
|---|---|---|
| PyTorch FP32 to ONNX FP32 export | 200/200 Top-1 prediction agreements; maximum absolute logit difference `1.5 × 10⁻⁵` | **Deployed** |
| YOLO26n detector INT8, detection head retained FP32 | Approximately 3.4 MB reported; source records disagree on the precise definition of the reported 0.988 box-agreement value; retained quantised artifact and exact benchmark details unavailable | Not deployed |
| YOLO26n full INT8, including detection head | Outcome not recorded / artifact unavailable | Not deployed |
| EfficientNetV2-S INT8, MinMax calibration | Accuracy collapsed to approximately 1% (rounded source value) | Not deployed |
| EfficientNetV2-S INT8, percentile calibration | Could not run due to out-of-memory on the 8 GB machine | Not deployed |

The deployed models remain FP32 ONNX. ONNX parity means agreement of measured Top-1 predictions on the 200 parity samples; it does not mean bit-for-bit identical logits.

## 8. Failure analysis

Failure analysis was conducted on the official test split after the configuration freeze. The source records contain a candidate pool of 479 entries, including 473 wrong Top-1 predictions, and 12 visually reviewed cases.

Examples include:

- Confusion between closely related model years and body styles (for example, Dodge Ram Crew Cab versus Quad Cab).
- Similar front-end appearance between coupe and sedan variants.
- Low-resolution images where grille, badge or trim details are not visible.
- Viewpoints that hide the feature distinguishing coupe and convertible variants.
- Visually similar badge-engineered vans.
- Detector/annotation mismatches in multi-vehicle or composite images.

The six category counts in the experiment record are overlapping tags, not a mutually exclusive partition of the candidate pool. The 12 reviewed cases are documented in the academic report and evaluation data. Dataset test images are not redistributed in the repository because of the dataset licence; references are provided for traceability.

Visual diagnoses are not necessarily experimentally proven causes. Proposed improvements in the failure analysis are not claimed as tested interventions.

---

## P5 requirement coverage and known gaps

| Requirement / capability | Evidence status |
|---|---|
| 196-class fine-grained classification | Evaluated on validation and official test |
| Vehicle detection and bounding-box localisation | YOLO26n/YOLO11n comparison and official-test IoU results |
| Transfer-learned deep classifier | EfficientNetV2-S and MobileNetV3-Large comparisons |
| Classical CV baseline | HOG, LBP and HSV with PCA and RBF-SVM |
| Classical edge detection, contour analysis and geometric features | Not demonstrated in the supplied evaluation evidence; verify implementation and add evidence before claiming completion |
| Classical/deep/hybrid comparison | Evaluated; fusion did not improve over deep-only |
| Confidence calibration and abstention | Temperature scaling and validation-selected rejection threshold evaluated |
| Synthetic robustness | Six perturbation families and three severity levels on validation ROIs |
| Multiple-vehicle interface | Vehicle selection is available in the interface; no dedicated multi-vehicle benchmark recorded |
| End-to-end latency | Not measured as one complete timed pipeline |
| Hosted cold/warm latency | Not measured |

Do not claim edge detection, contour analysis or geometric shape descriptors as implemented unless the relevant source code and application output are present. If these features are implemented, add a screenshot or code reference and sample computed features to the project documentation.

---

## Repository layout

```text
.
├── app.py
├── src/
│   └── vehicle_recognition/
│       ├── data/
│       ├── preprocessing/
│       ├── classical_cv/
│       ├── detection/
│       ├── classification/
│       ├── calibration/
│       ├── fusion/
│       ├── evaluation/
│       ├── inference/
│       └── visualisation/
├── scripts/
│   ├── data acquisition and validation
│   ├── training and evaluation
│   ├── export and benchmarking
│   └── report generation
├── configs/
├── notebooks/
│   ├── kaggle/
│   └── colab/
├── reports/
│   ├── dataset/
│   ├── results/
│   └── ACADEMIC_REPORT.md
├── tests/
├── data/
│   └── README.md
├── .github/
│   └── workflows/
├── Dockerfile
├── Makefile
├── pyproject.toml
└── uv.lock
```

The tree above describes the documented project structure. Consult the repository for the current exact file and directory inventory.

---

## Run locally

### Requirements

- Python 3.11
- Dependencies declared in `uv.lock`

### Install and launch

```bash
python3.11 -m pip install --user uv
uv sync --extra research --extra dev
make test
make app
```

The Streamlit application is served locally at:

```text
http://localhost:8501
```

The pinned model bundle is downloaded on first start, according to the deployment configuration.

### Dataset and experiment reproduction

```bash
make data validate-data
```

Then run the relevant Kaggle or Colab notebook in `notebooks/` and the evaluation scripts in `scripts/`. The notebooks download the original dataset themselves and are designed to support resumable training.

Avoid placing the project in an iCloud-synchronised folder. See [`data/README.md`](data/README.md) for dataset setup details.

---

## Docker

Build the image:

```bash
docker build -t vehicle-recognition .
```

Run the container:

```bash
docker run --rm -p 7860:7860 vehicle-recognition
```

The Docker configuration is documented as using runtime dependencies without PyTorch, running as non-root UID 1000, bounding runtime threads and including a health check. The repository reports that GitHub Actions builds and smoke-tests the Linux x86_64 image on each push through [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

Check the Dockerfile's exposed/listening port and health-check configuration if changing the container run command.

---

## Limitations

- The classifier recognises only the 196 Stanford Cars classes, which largely cover 1991–2012 US-market vehicles. Newer or unsupported vehicles may be assigned to the nearest known class; the system does not establish that such a prediction is correct. Low confidence may trigger abstention.
- Multi-vehicle scenes can be handled through the interface, but dedicated multi-vehicle benchmark results are not available.
- Robustness measurements use synthetic perturbations on validation ROIs, not naturally collected adverse-condition images.
- A single end-to-end inference latency measurement and hosted Streamlit cold/warm latency were not recorded.
- Official-test detector mAP was not measured.
- The 95% selective-accuracy target was met on validation but not on the official test.
- The classical/deep fusion experiment did not improve on the deep-only classifier.
- Some INT8 experiments were unsuccessful or incompletely retained; the deployed models remain FP32 ONNX.
- The application does not perform number-plate recognition, vehicle tracking, damage assessment, valuation, ownership identification or surveillance.

---

## Dataset, citation and licence

### Dataset

Stanford Cars is used for research and is **not redistributed** in this repository. Users must obtain the original dataset files themselves and follow the instructions in [`data/README.md`](data/README.md).

### Citation

> J. Krause, M. Stark, J. Deng, L. Fei-Fei. “3D Object Representations for Fine-Grained Categorization.” 4th IEEE Workshop on 3D Representation and Recognition (3dRR-13), ICCV 2013.

### Licence

- **Code:** AGPL-3.0-only, as stated by the project.
- **Trained weights:** designated for non-commercial research and education only.

Review [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for third-party notices and applicable terms. Ensure the licence files in the repository agree with these statements before distribution.

---

<div align="center">

**P5 · Team T15 — Pixel**

*Fine-grained vehicle recognition through classical computer vision, deep learning and systematic evaluation.*

</div>
