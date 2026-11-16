# HybriDet

**HybriDet: A Hybrid Neural Network Combining CNN and Transformer for Wildfire Detection in Remote Sensing Imagery**

[Paper (Remote Sensing 2025)](https://doi.org/10.3390/rs17203497) · [GitHub](https://github.com/dfm021101/HybriDet) · [中文说明](README.zh-CN.md)

**Authors:** Fengming Dong · Ming Wang (corresponding: oyamingo@whu.edu.cn)

HybriDet is a lightweight hybrid detector that combines CNN local modeling with Transformer global context for wildfire / smoke detection in remote sensing imagery. It is built on **Ultralytics YOLOv8 (`8.2.14`)** and introduces:

- **SwinBottle** — Swin Transformer + bottleneck residual paths for efficient global–local feature extraction
- **CS Attention** — Coordinate–Spatial dual attention for irregular flame / smoke targets
- **SliceSamp / SliceUpsamp** — parameter-efficient down / upsampling
- **Concat_BiFPN** — weighted multi-scale feature fusion

After structured pruning, the lightweight variant has **6.45 M** parameters and improves **mAP50 by 6.4%** over YOLOv8 on FASDD-RS, while remaining suitable for edge deployment.

---

## Environment

Requirements are defined in [`pyproject.toml`](pyproject.toml).

| Item | Requirement |
| --- | --- |
| Python | `>= 3.8` (tested classifiers: 3.8–3.12) |
| PyTorch | `torch >= 1.8.0` |
| TorchVision | `torchvision >= 0.9.0` |
| OS | Linux / macOS / Windows |
| GPU (recommended) | NVIDIA CUDA GPU for training |

### Core dependencies (`pyproject.toml`)

```
matplotlib>=3.3.0
opencv-python>=4.6.0
pillow>=7.1.2
pyyaml>=5.3.1
requests>=2.23.0
scipy>=1.4.1
torch>=1.8.0
torchvision>=0.9.0
tqdm>=4.64.0
psutil
py-cpuinfo
thop>=0.1.1
pandas>=1.1.4
seaborn>=0.11.0
```

Optional extras in the same file: `dev`, `export`, `explorer`, `logging`, `extra`.

---

## Installation

> **Important:** install from this repository in editable mode. Do **not** rely on `pip install ultralytics` from PyPI alone — that package does not include HybriDet modules (`SwinBottle`, `CS_Attention`, etc.).

### 1) Create a virtual environment (recommended)

```bash
conda create -n hybridet python=3.10 -y
conda activate hybridet
# or: python -m venv .venv && source .venv/bin/activate
```

### 2) Install PyTorch (GPU / CPU)

Pick a build that matches your CUDA driver. Examples aligned with this repo’s Dockerfiles:

```bash
# CUDA 12.1 (see docker/Dockerfile: pytorch/pytorch:2.2.2-cuda12.1-...)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Or CUDA 11.8 (see docker/Dockerfile-conda)
# conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# CPU-only
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3) Install HybriDet (this repo)

```bash
cd HybriDet-main
pip install -U pip wheel
pip install -e .
```

Optional feature sets:

```bash
pip install -e ".[export]"   # ONNX / OpenVINO / CoreML / TF ...
pip install -e ".[logging]"  # TensorBoard / Comet / DVCLive
pip install -e ".[extra]"    # albumentations, pycocotools, ...
```

### 4) Verify

```bash
python -c "from ultralytics import YOLO; print('ok', YOLO)"
yolo checks
```

### Docker (optional)

```bash
# GPU image base used in-repo: pytorch 2.2.2 + CUDA 12.1
docker build -f docker/Dockerfile -t hybridet:latest .
docker run -it --ipc=host --gpus all -v "$(pwd)":/workspace hybridet:latest
```

Also available: `docker/Dockerfile-cpu`, `docker/Dockerfile-conda`, `docker/Dockerfile-arm64`, `docker/Dockerfile-jetson`.

---

## Quick Start

### Download the FASDD dataset

The main paper experiments use [FASDD](https://github.com/openrsgis/FASDD) (Flame And Smoke Detection Dataset). Images are hosted on Science Data Bank — **`git clone` alone does not provide the data**.

| Item | Detail |
| --- | --- |
| Project page | https://github.com/openrsgis/FASDD |
| Dataset DOI | https://doi.org/10.57760/sciencedb.j00104.00103 |
| ScienceDB / SciDB | https://www.scidb.cn/en/detail?dataSetId=ce9c9400b44148e1b0a749f5c3eb0bda |
| YOLO classes | `0: fire`, `1: smoke` |
| Official split | train : val : test ≈ **1/2 : 1/3 : 1/6** |

**Which archive to download:**

| Archive | Content | Approx. size | Notes |
| --- | --- | --- | --- |
| `FASDD_RS.zip` | Satellite RS imagery | ~2.2k images | Smallest; start here for paper RS runs |
| `FASDD_UAV.zip` | UAV imagery | ~25k images | UAV experiments |
| `FASDD_CV.zip` | General CV scenes | ~95k images | Largest |
| `FASDD_RS_SWIR.zip` | RS SWIR pseudo-color | — | Optional for SWIR flame detection |

**Steps:**

1. Open the DOI / SciDB page, sign in, and download the zip you need (recommend `FASDD_RS.zip` first).
2. Extract locally, e.g.:

```bash
mkdir -p datasets && cd datasets
unzip FASDD_RS.zip -d FASDD_RS
# Typical layout (follow the official release):
# FASDD_RS/
#   images/
#   annotations/
#     YOLO/   # or YOLO_RS, etc.
#       labels/train|val|test
```

3. Reorganize into the Ultralytics YOLO layout if needed (`images/{train,val,test}` + `labels/{train,val,test}` with matching filenames). Each YOLO label line is:

```text
class_id x_center y_center width height   # normalized by image width/height
```

4. Create a data YAML (not shipped in this repo), e.g. `data/fasdd_rs.yaml`:

```yaml
path: /path/to/datasets/FASDD_RS_yolo
train: images/train
val: images/val
test: images/test  # optional

names:
  0: fire
  1: smoke
```

5. Train HybriDet with that YAML:

```bash
yolo detect train \
  model=ultralytics/cfg/models/v8/HybriDet.yaml \
  data=data/fasdd_rs.yaml \
  imgsz=640 epochs=1000 batch=64 device=0
```

Please cite the FASDD paper (Wang et al., Geo-spatial Information Science, 2025) when using the dataset.

Pascal VOC is also used in the paper; prepare it in YOLO format the same way.

### Train

Edit absolute paths / GPU id in [`train.py`](train.py), then:

```bash
python train.py
```

Or call the API directly (settings match the example script):

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/v8/HybriDet.yaml")
model.train(
    data="data/fasdd_rs.yaml",  # your dataset yaml
    imgsz=640,
    epochs=1000,
    batch=64,
    workers=8,
    device=0,                  # or "0,1" / "cpu"
    project="HybriDet",
    name="exp",
    save=True,
    save_period=1,
    resume=False,
)
```

Model config: [`ultralytics/cfg/models/v8/HybriDet.yaml`](ultralytics/cfg/models/v8/HybriDet.yaml)

Default Ultralytics hyperparameters live in [`ultralytics/cfg/default.yaml`](ultralytics/cfg/default.yaml) (`batch=16`, `epochs=100`, AMP on, etc.) and are overridden by arguments you pass above.

### Validate / Predict

```python
from ultralytics import YOLO

model = YOLO("runs/detect/exp/weights/best.pt")
metrics = model.val(data="data/fasdd_rs.yaml")
results = model.predict(source="path/to/image_or_dir", save=True)
```

CLI:

```bash
yolo detect val model=runs/detect/exp/weights/best.pt data=data/fasdd_rs.yaml
yolo detect predict model=runs/detect/exp/weights/best.pt source=path/to/images
```

---

## Model Overview

| Component | Role |
| --- | --- |
| Backbone | `SliceSamp` + `SwinBottle` + `SPPF` |
| Neck | `C2f` + `CS_Attention` + `Concat_BiFPN` + `SliceUpsamp` / `SliceSamp` |
| Head | YOLOv8-style decoupled Detect head (P3 / P4 / P5) |

Key files:

- `ultralytics/nn/modules/block.py` — `SwinBottle`, `CS_Attention`, `SliceSamp`, `Concat_BiFPN`, …
- `ultralytics/cfg/models/v8/HybriDet.yaml` — network definition
- `train.py` — example training script
- `pyproject.toml` — environment / dependency definition

---

## Core Module Code Guide

Both core modules live in [`ultralytics/nn/modules/block.py`](ultralytics/nn/modules/block.py), are exported from [`ultralytics/nn/modules/__init__.py`](ultralytics/nn/modules/__init__.py), and are wired in [`HybriDet.yaml`](ultralytics/cfg/models/v8/HybriDet.yaml).

### 1. SwinBottle (backbone feature extraction)

| Item | Location |
| --- | --- |
| Class | `block.py` → `SwinBottle` (≈ L328–347) |
| Depends on | same file: `SwinTransformerBlock` (≈ L50–66), `SwinTransformerLayer` / `WindowAttention` |
| YAML usage | Backbone: `[-1, n, SwinBottle, [c2, True]]` |
| Parser | `ultralytics/nn/tasks.py` (parsed like `C2f`) |

**Data flow:** input → `SwinTransformerBlock(c1 → 2c)` → `chunk(2)` → `Bottleneck` on one branch → `concat` → `1×1 Conv` → `c2`.

```328:347:ultralytics/nn/modules/block.py
class SwinBottle(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        ...
        self.c = int(c2 * e)  # hidden channels = c2/2
        self.transformer = SwinTransformerBlock(c1, 2 * self.c, 2 * self.c // 16, n)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.transformer(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
```

YAML sites (`HybriDet.yaml` backbone):

```yaml
- [-1, 3, SwinBottle, [128, True]]
- [-1, 6, SwinBottle, [256, True]]
- [-1, 6, SwinBottle, [512, True]]
- [-1, 3, SwinBottle, [1024, True]]
```

Suggested reading order: `SwinBottle` → `SwinTransformerBlock` → `SwinTransformerLayer` / `WindowAttention`.

### 2. CS Attention (neck Coordinate–Spatial dual attention)

| Item | Location |
| --- | --- |
| Class | `block.py` → `CS_Attention` (≈ L422–447) |
| Channel attn | same file: `CoordAtt` (≈ L384–419), with `h_swish` / `h_sigmoid` |
| Spatial attn | `CS_Attention.spatial_attention`: two `7×7 Conv` + BN + ReLU |
| YAML usage | Neck: `[-1, 3, CS_Attention, [c]]` |

**Data flow:** `CoordAtt` (H/W pooling → channel attention) → BN → multiply → `spatial_attention` → sigmoid → multiply.

```422:447:ultralytics/nn/modules/block.py
class CS_Attention(nn.Module):
    def __init__(self, in_channels, rate=2):
        ...
        self.channel_attention = CoordAtt(in_channels, in_channels)
        self.batchnorm = nn.BatchNorm2d(in_channels)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
            ...
            nn.Conv2d(int(in_channels / rate), in_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        x_channel_att = self.channel_attention(x)
        x_channel_att = self.batchnorm(x_channel_att)
        x = x * x_channel_att
        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att
        return out
```

YAML sites (`HybriDet.yaml` neck):

```yaml
- [-1, 3, CS_Attention, [512]]
- [-1, 3, CS_Attention, [256]]
- [-1, 3, CS_Attention, [512]]
- [-1, 3, CS_Attention, [1024]]
```

Suggested reading order: `CS_Attention` → `CoordAtt` → `h_swish` / `h_sigmoid`.

---

## Clarifications (FAQ)

### 1. SwinBottle channel split ratio

**Question.** Section 3.2 of the paper states that the feature map is split into “three feature tensors of identical dimensions.” Table 1 channel widths `[32, 64, 128, 256]` are not divisible by 3. Parameter counting suggests `[C/2, C/4, C/4]` matches Table 1 much better than `[C/3, C/3, C/3]`. Which split is used in the implementation?

**Answer.** The released code does **not** split channels as `[C/3, C/3, C/3]` or `[C/2, C/4, C/4]`.

`SwinBottle` follows the **C2f-style** design (`ultralytics/nn/modules/block.py`):

1. Expansion ratio `e = 0.5`, so hidden width `c = C / 2` (where `C` is the stage output channels, e.g. 32 / 64 / 128 / 256).
2. The Swin Transformer block maps the input to `2c = C` channels.
3. Those `C` channels are split with `chunk(2, 1)` into **two equal halves** of size `C/2`.
4. The third path is produced by a `Bottleneck` on the second half, also with width `C/2`.
5. The three tensors (for `n = 1`) are therefore **identical in shape**: `[C/2, C/2, C/2]`, then concatenated and projected by a `1×1` conv.

So “three feature tensors of identical dimensions” means three branches of width **`C/2`** after expand-and-split, **not** a direct `C/3` partition of `C`.

```python
# SwinBottle forward (simplified)
self.c = int(c2 * 0.5)                          # C/2
self.transformer = SwinTransformerBlock(..., 2 * self.c, ...)
y = list(self.transformer(x).chunk(2, 1))      # [C/2, C/2]
y.extend(m(y[-1]) for m in self.m)             # + Bottleneck → C/2
return self.cv2(torch.cat(y, 1))
```

---

## Citation

```bibtex
@article{dong2025hybridet,
  title   = {HybriDet: A Hybrid Neural Network Combining CNN and Transformer for Wildfire Detection in Remote Sensing Imagery},
  author  = {Dong, Fengming and Wang, Ming},
  journal = {Remote Sensing},
  volume  = {17},
  number  = {20},
  pages   = {3497},
  year    = {2025},
  doi     = {10.3390/rs17203497}
}
```

---

## Acknowledgements

Built on [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics). Please cite Ultralytics when using the upstream framework, and cite HybriDet (Dong & Wang, *Remote Sensing* 2025) for this method. Thanks to the Ultralytics team and the FASDD dataset authors.

Preferred software citation metadata: [`CITATION.cff`](CITATION.cff).

## License

See [LICENSE](LICENSE). Code derived from Ultralytics follows AGPL-3.0 unless otherwise noted.
