# Third-Party Notices

This project is licensed under **AGPL-3.0-only** (see `LICENSE`). That choice is required by the Ultralytics
YOLO code and weights used to train and export the vehicle detector. Licences were checked on 2026-11-06 from
installed package metadata (`importlib.metadata`) and official project pages.

## Dataset (not redistributed)

| Item | Terms | How it is used here |
|---|---|---|
| Stanford Cars (Krause, Stark, Deng, Fei-Fei, 2013) | "This dataset is for research purposes only, similar to the ImageNet license." (official dataset page, archived capture) | Downloaded by each user from the original files (see `data/README.md`). **No images, annotations or per-image labels are committed** to this repository or included in the Docker image or the deployed Space. Only class names (196 make/model/year strings) and aggregate statistics are included. |

Citation: J. Krause, M. Stark, J. Deng, L. Fei-Fei. *3D Object Representations for Fine-Grained Categorization.*
4th IEEE Workshop on 3D Representation and Recognition, at ICCV 2013 (3dRR-13), Sydney, Australia.

**Trained model weights** in the deployment bundle were fine-tuned on Stanford Cars. They are published for
**non-commercial research and educational use only**, consistent with the dataset terms. The repository owner
approved this on 2026-11-06.

## Pretrained weights used for initialisation

| Weights | Licence / terms | Notes |
|---|---|---|
| Ultralytics YOLO26n, YOLO11n (`yolo26n.pt`, `yolo11n.pt`, COCO-pretrained) | AGPL-3.0 (Ultralytics); Enterprise licence needed for closed-source/commercial use | Fine-tuned detector weights are derivative works and are therefore AGPL-3.0. |
| torchvision ImageNet weights (MobileNetV3-Large `IMAGENET1K_V2`, EfficientNetV2-S `IMAGENET1K_V1`) | torchvision code is BSD-3-Clause; the weights were trained on ImageNet-1k, whose terms restrict use to non-commercial research/education | Used only as initialisation for non-commercial research. |

## Python dependencies

Versions are those in `uv.lock` at the time of writing.

| Package | Version | Licence | Used in |
|---|---|---|---|
| numpy | 2.4.6 | BSD-3-Clause (+ bundled 0BSD/MIT/Zlib/CC0) | runtime |
| Pillow | 13.0.0 | MIT-CMU (HPND) | runtime |
| opencv-python-headless | 5.0.0.93 | Apache-2.0 | runtime |
| scikit-image | 0.26.0 | BSD-3-Clause | runtime |
| scikit-learn | 1.9.1 | BSD-3-Clause | runtime |
| joblib | 1.6.0 | BSD-3-Clause | runtime |
| onnxruntime | 1.31.1 | MIT | runtime |
| streamlit | 1.67.0 | Apache-2.0 | runtime |
| plotly | 7.1.0 | MIT | runtime |
| pandas | 3.1.1 | BSD-3-Clause | runtime |
| PyYAML | 6.0.3 | MIT | runtime |
| pydantic | 2.14.3 | MIT | runtime |
| psutil | 7.2.2 | BSD-3-Clause | runtime |
| scipy | 1.17.1 | BSD-3-Clause | research |
| torch | 2.15.0 | BSD-3-Clause-style (+ Apache-2.0, MIT, etc. for bundled components) | research/training |
| torchvision | 0.29.2 | BSD-3-Clause | research/training |
| ultralytics | 8.4.203 | AGPL-3.0 | research/training/export |
| onnx | 1.23.2 | Apache-2.0 | export |
| onnxslim | 0.1.99 | MIT | export |
| ImageHash | 4.3.2 | BSD-2-Clause | dataset audit |
| matplotlib | 3.11.2 | PSF-based (matplotlib licence) | reports |
| pyarrow | 26.0.1 | Apache-2.0 | metadata I/O |

All of the above are compatible with distributing this project under AGPL-3.0.
