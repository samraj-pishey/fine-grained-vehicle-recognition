# Intelligent Fine-Grained Vehicle Recognition and Analysis System

Upload a photo of a car. The system **detects** the vehicle(s), lets you **select** one, and **classifies** it
into one of the **196 Stanford Cars make/model/year classes**. It reports calibrated confidence and the Top-5
alternatives, and **abstains** when it is not confident enough. The project is also a research comparison of
classical computer vision (HOG/LBP/HSV + calibrated SVM), deep learning (MobileNetV3-Large vs EfficientNetV2-S),
and hybrid probability fusion.

**Live demo:** https://vehicle-recognition.streamlit.app (free Streamlit Community Cloud; the first visit after a pause may take a minute to wake up) · **Model bundle:**
[release `models-v1`](https://github.com/samraj-pishey/fine-grained-vehicle-recognition/releases/tag/models-v1)

> Every number below was produced by a recorded experiment in `reports/results/`. *Validation* numbers come from a
> stratified, duplicate-grouped split of the **official training set**. Official-test numbers are reported only in
> [`reports/ACADEMIC_REPORT.md`](reports/ACADEMIC_REPORT.md), computed **once** after all choices were frozen.
> Progress is tracked in [`milestone_acceptance.yaml`](milestone_acceptance.yaml).

## How it works

```text
upload ─► content validation (JPEG/PNG signature, ≤10 MB, ≤25 MP, full decode) ─► EXIF fix ─► RGB
      ─► YOLO26n vehicle detector (ONNX Runtime, CPU) ─► numbered boxes ─► user selects a vehicle
      ─► ROI crop (+5 % pad) ─► EfficientNetV2-S, 288 px (ONNX) ─► temperature scaling (T = 0.77)
      ─► abstain if confidence < 0.397 ─► Top-5 + make / model / year (official class names)
```

## Results (validation, 1,215 images from the official train split)

**Detector**: single-class fine-tuning on Stanford Cars boxes, identical settings for both candidates.

| Model | mAP50 | mAP50-95 | Miss rate (IoU < 0.5) | Median IoU | CPU ms (M1, 2 threads) |
|---|---|---|---|---|---|
| **YOLO26n (selected)** | 0.995 | 0.939 | 0.08 % | 0.969 | **53** |
| YOLO11n | 0.995 | 0.940 | 0.08 % | 0.966 | 66 |

The two were equally accurate, so the pre-declared rule chose the faster one.

**Classifiers**: ground-truth crops, ImageNet-pretrained, 40 epochs on a free Kaggle T4.

| Run | Top-1 | Top-5 | Macro-F1 | ECE before → after calibration | CPU ms |
|---|---|---|---|---|---|
| **EfficientNetV2-S @288 (selected)** | **94.3 %** | 98.8 % | 0.942 | 0.099 → 0.028 | 74 |
| EfficientNetV2-S @224 | 92.8 % | 98.9 % | 0.927 | 0.095 → 0.031 | 48 |
| MobileNetV3-L @288 | 90.7 % | 97.9 % | 0.903 | 0.301 → 0.034 | 14 |
| MobileNetV3-L @224 (3 seeds) | 89.4 ± 0.25 % | 97.6 % | 0.89 | 0.31 → 0.04 | 10 |
| MobileNetV3-L @224, light augmentation | 87.4 % | 97.0 % | 0.872 | 0.345 → 0.051 | – |
| MobileNetV3-L @224, class-weighted loss | 89.5 % | 97.5 % | 0.891 | 0.315 → 0.047 | – |

The calibrated ECE values are cross-fitted on validation halves.

**Classical branch** (HOG + uniform LBP + HSV → standardisation → PCA 95 % → RBF-SVM with Platt calibration):
9.4 % top-1, 31.8 % top-5. HOG carries most of the signal; the full ablation is in
`reports/results/classical_ablation.json`. **Hybrid fusion** P = α·P_deep + (1−α)·P_classical was evaluated on
validation. The best α was 1.0 (no gain), so production runs **deep-only**, as the pre-declared rule requires.

**End-to-end** (detector box → classifier, validation): correct box and correct class for **93.9 %** of images.
Using the detector's box instead of the ground-truth box changes the prediction for 1.0 % of images.

**Abstention**: the threshold was chosen on validation so that the answers given are ≥ 95 % correct. Cross-fitted:
98.9 % of images are answered, with 95.2 % of those correct.

**Optimisation**: the ONNX export matches PyTorch exactly (200/200 identical top-1; max logit difference 1.5e-5).
Static INT8 quantisation collapsed the EfficientNetV2-S classifier to 1 % accuracy, so the deployed model is FP32.
The INT8 detector works when its detection head is kept in FP32 (3.4 MB vs 9.9 MB; boxes 98.8 % IoU-identical).

**Dataset audit** (`reports/dataset/`): counts match the official release (8,144 train / 8,041 test / 196 classes).
The archives are byte-identical to the original Stanford files (MD5 matches torchvision's pins). 24 train↔test
duplicates were removed from training. **44 duplicate photo pairs carry conflicting labels** in the original data
(for example Dodge Caliber 2007 vs 2012), which is label noise in the source.

## Repository layout

```text
app.py                      Streamlit entrypoint (tabs: recognition, detection/ROI, classical CV, deep vs hybrid,
                            performance, robustness & failures, methodology)
src/vehicle_recognition/    data · preprocessing · classical · detection · classification · calibration ·
                            fusion · evaluation · inference · visualization · app
scripts/                    acquisition, validation, training, evaluation, export/benchmark, report generation
configs/                    dataset sources (SHA-256), training plan, classical, deployment (pinned bundle)
notebooks/{kaggle,colab}/   GPU training notebooks (download the data themselves; resumable)
reports/                    dataset audit, experiment results, academic report
tests/                      unit, integration (real ONNX), headless app end-to-end tests
```

## Run locally

```bash
python3.11 -m pip install --user uv
uv sync --extra research --extra dev        # reproducible environment from uv.lock
make test                                   # unit + integration + app tests
make app                                    # http://localhost:8501; downloads the pinned model bundle on first start
```

To reproduce the data and experiments: `make data validate-data`, then run the Kaggle/Colab notebook in `notebooks/`
(it downloads the original files and trains the full plan), and then the evaluation scripts in `scripts/`. Keep the
project out of iCloud-synced folders; see `data/README.md`.

## Docker

```bash
docker build -t vehicle-recognition . && docker run -p 7860:7860 vehicle-recognition
```

The image is slim (runtime dependencies only, no PyTorch), runs as non-root uid 1000, bounds its threads and has
a health check. It is built and smoke-tested on Linux x86_64 by GitHub Actions on every push (`.github/workflows/ci.yml`).

## Limitations

- Recognises only the 196 Stanford Cars classes, mostly 1991–2012 US-market models. Newer cars are out of scope
  and are mapped to the nearest known class or flagged as uncertain.
- The detector was trained on photos with one annotated car each. Multi-vehicle scenes work in the interface but
  are not benchmarked.
- Robustness is measured with synthetic perturbations, not natural conditions.
- On free hosting the app sleeps when idle, so the first visit after a pause is slow while it starts and loads models.
- Does **not** do number plates, tracking, damage, valuation, ownership or surveillance.

## Dataset, citation and licence

Stanford Cars is **research-only** and is not redistributed here. Each user downloads the original files
(`data/README.md`).

> J. Krause, M. Stark, J. Deng, L. Fei-Fei. *3D Object Representations for Fine-Grained Categorization.*
> 4th IEEE Workshop on 3D Representation and Recognition (3dRR-13), ICCV 2013.

Code: **AGPL-3.0-only** (required by Ultralytics YOLO). Trained weights: **non-commercial research and education
only**. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
