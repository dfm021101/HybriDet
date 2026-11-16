# HybriDet

**HybriDet：面向遥感野火检测的 CNN–Transformer 混合神经网络**

[论文（Remote Sensing 2025）](https://doi.org/10.3390/rs17203497) · [GitHub](https://github.com/dfm021101/HybriDet) · [English](README.md)

**作者：** 董丰铭（Fengming Dong）· 王明（Ming Wang，通讯作者：oyamingo@whu.edu.cn）

HybriDet 是面向遥感影像野火 / 烟雾检测的轻量混合检测器，基于 **Ultralytics YOLOv8（`8.2.14`）** 开发，融合 CNN 局部建模与 Transformer 全局上下文，主要模块包括：

- **SwinBottle**：Swin Transformer + Bottleneck 残差路径，高效提取全局–局部特征
- **CS Attention**：坐标–空间双重注意力，增强不规则火焰 / 烟雾目标判别
- **SliceSamp / SliceUpsamp**：参数更省的下采样 / 上采样
- **Concat_BiFPN**：加权多尺度特征融合

结构化剪枝后的轻量版本约 **6.45 M** 参数，在 FASDD-RS 上相对 YOLOv8 的 **mAP50 提升 6.4%**，适合边缘端部署。

---

## 环境要求

依赖定义见 [`pyproject.toml`](pyproject.toml)。

| 项目 | 要求 |
| --- | --- |
| Python | `>= 3.8`（声明支持 3.8–3.12） |
| PyTorch | `torch >= 1.8.0` |
| TorchVision | `torchvision >= 0.9.0` |
| 操作系统 | Linux / macOS / Windows |
| GPU（推荐） | NVIDIA CUDA，用于训练加速 |

### 核心依赖（`pyproject.toml`）

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

可选扩展同文件中的 extras：`dev`、`export`、`explorer`、`logging`、`extra`。

---

## 安装

> **注意：** 请从本仓库以可编辑模式安装。不要只执行 PyPI 的 `pip install ultralytics`——官方包不包含 HybriDet 的 `SwinBottle`、`CS_Attention` 等模块。

### 1）创建虚拟环境（推荐）

```bash
conda create -n hybridet python=3.10 -y
conda activate hybridet
# 或: python -m venv .venv && source .venv/bin/activate
```

### 2）安装 PyTorch（GPU / CPU）

请按本机 CUDA 驱动选择对应版本。可与本仓库 Docker 配置对齐：

```bash
# CUDA 12.1（见 docker/Dockerfile: pytorch/pytorch:2.2.2-cuda12.1-...）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 或 CUDA 11.8（见 docker/Dockerfile-conda）
# conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# 仅 CPU
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3）安装本仓库（HybriDet）

```bash
cd HybriDet-main
pip install -U pip wheel
pip install -e .
```

可选功能包：

```bash
pip install -e ".[export]"   # ONNX / OpenVINO / CoreML / TF 等
pip install -e ".[logging]"  # TensorBoard / Comet / DVCLive
pip install -e ".[extra]"    # albumentations、pycocotools 等
```

### 4）验证安装

```bash
python -c "from ultralytics import YOLO; print('ok', YOLO)"
yolo checks
```

### Docker（可选）

```bash
# 仓库 GPU 镜像基座：PyTorch 2.2.2 + CUDA 12.1
docker build -f docker/Dockerfile -t hybridet:latest .
docker run -it --ipc=host --gpus all -v "$(pwd)":/workspace hybridet:latest
```

另有：`docker/Dockerfile-cpu`、`docker/Dockerfile-conda`、`docker/Dockerfile-arm64`、`docker/Dockerfile-jetson`。

---

## 快速开始

### 下载 FASDD 数据集

论文主实验数据为 [FASDD](https://github.com/openrsgis/FASDD)（Flame And Smoke Detection Dataset）。数据托管在科学数据银行，**不能**仅靠 `git clone` 拿到图像。

| 项目 | 说明 |
| --- | --- |
| 项目页 | https://github.com/openrsgis/FASDD |
| 数据 DOI | https://doi.org/10.57760/sciencedb.j00104.00103 |
| ScienceDB / SciDB | https://www.scidb.cn/en/detail?dataSetId=ce9c9400b44148e1b0a749f5c3eb0bda |
| 类别（YOLO） | `0: fire`，`1: smoke` |
| 官方划分 | train : val : test ≈ **1/2 : 1/3 : 1/6** |

**子集选择（按需下载）：**

| 压缩包 | 内容 | 规模（约） | 建议 |
| --- | --- | --- | --- |
| `FASDD_RS.zip` | 遥感卫星影像 | ~2.2k 张 | 论文 RS 实验，体积最小，优先下载 |
| `FASDD_UAV.zip` | 无人机影像 | ~25k 张 | UAV 实验 |
| `FASDD_CV.zip` | 通用视觉场景 | ~95k 张 | 最大，磁盘与训练成本高 |
| `FASDD_RS_SWIR.zip` | RS 短波红外伪彩 | — | SWIR 火焰检测可选 |

**下载步骤：**

1. 打开 DOI 或 SciDB 页面，注册 / 登录后下载所需 zip（推荐先下 `FASDD_RS.zip`）。
2. 解压到本地数据目录，例如：

```bash
mkdir -p datasets && cd datasets
# 将浏览器下载的 FASDD_RS.zip 放到当前目录后：
unzip FASDD_RS.zip -d FASDD_RS
# 典型结构（以官方发布为准）：
# FASDD_RS/
#   images/                  # 原始图像
#   annotations/
#     YOLO/                  # 或 YOLO_RS 等命名
#       labels/train|val|test
#       (以及划分列表 / 对应标签)
```

3. 整理为 Ultralytics YOLO 目录布局（若解压后已是 `images/{train,val,test}` + `labels/{train,val,test}` 可跳过）。常用做法是把官方 YOLO 标签与图像按 split 对齐：

```bash
# 示例目标结构
# datasets/FASDD_RS_yolo/
#   images/train  images/val  images/test
#   labels/train  labels/val  labels/test
```

保证 **同名** 图像与 `.txt` 标签一一对应；YOLO 标签每行为：

```text
class_id x_center y_center width height   # 均为相对图像宽高的归一化坐标
```

4. 编写数据配置 YAML（仓库未附带 datasets yaml，需自建），例如 `data/fasdd_rs.yaml`：

```yaml
path: /path/to/datasets/FASDD_RS_yolo
train: images/train
val: images/val
test: images/test  # 可选

names:
  0: fire
  1: smoke
```

5. 用该 YAML 训练 HybriDet：

```bash
yolo detect train \
  model=ultralytics/cfg/models/v8/HybriDet.yaml \
  data=data/fasdd_rs.yaml \
  imgsz=640 epochs=1000 batch=64 device=0
```

**引用数据集时请同时引用 FASDD 论文**（Wang et al., Geo-spatial Information Science, 2025）。

其他可用数据：Pascal VOC（同样需自行准备 YOLO 格式与 YAML）。

### 训练

请先修改 [`train.py`](train.py) 中的绝对路径与 GPU 编号，再运行：

```bash
python train.py
```

或直接调用 API（参数与示例脚本一致）：

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/v8/HybriDet.yaml")
model.train(
    data="data/fasdd_rs.yaml",  # 你的数据集配置
    imgsz=640,
    epochs=1000,
    batch=64,
    workers=8,
    device=0,                   # 或 "0,1" / "cpu"
    project="HybriDet",
    name="exp",
    save=True,
    save_period=1,
    resume=False,
)
```

模型配置：[`ultralytics/cfg/models/v8/HybriDet.yaml`](ultralytics/cfg/models/v8/HybriDet.yaml)

Ultralytics 默认超参见 [`ultralytics/cfg/default.yaml`](ultralytics/cfg/default.yaml)（如 `batch=16`、`epochs=100`、默认开启 AMP），会被你传入的参数覆盖。

### 验证 / 推理

```python
from ultralytics import YOLO

model = YOLO("runs/detect/exp/weights/best.pt")
metrics = model.val(data="data/fasdd_rs.yaml")
results = model.predict(source="path/to/image_or_dir", save=True)
```

命令行：

```bash
yolo detect val model=runs/detect/exp/weights/best.pt data=data/fasdd_rs.yaml
yolo detect predict model=runs/detect/exp/weights/best.pt source=path/to/images
```

---

## 模型结构概览

| 组件 | 作用 |
| --- | --- |
| Backbone | `SliceSamp` + `SwinBottle` + `SPPF` |
| Neck | `C2f` + `CS_Attention` + `Concat_BiFPN` + `SliceUpsamp` / `SliceSamp` |
| Head | YOLOv8 风格解耦 Detect 头（P3 / P4 / P5） |

关键文件：

- `ultralytics/nn/modules/block.py` — `SwinBottle`、`CS_Attention`、`SliceSamp`、`Concat_BiFPN` 等
- `ultralytics/cfg/models/v8/HybriDet.yaml` — 网络定义
- `train.py` — 训练示例
- `pyproject.toml` — 环境与依赖定义

---

## 核心模块代码指引

两个核心模块均在 [`ultralytics/nn/modules/block.py`](ultralytics/nn/modules/block.py)，经 [`ultralytics/nn/modules/__init__.py`](ultralytics/nn/modules/__init__.py) 导出，并在 [`HybriDet.yaml`](ultralytics/cfg/models/v8/HybriDet.yaml) 中组网。

### 1. SwinBottle（Backbone 特征提取）

| 项目 | 位置 |
| --- | --- |
| 类定义 | `block.py` → `SwinBottle`（约 L328–347） |
| 依赖 | 同文件 `SwinTransformerBlock`（约 L50–66）、`SwinTransformerLayer` / `WindowAttention` |
| YAML 用法 | Backbone：`[-1, n, SwinBottle, [c2, True]]` |
| 解析注册 | `ultralytics/nn/tasks.py`（`SwinBottle` 与 `C2f` 同类解析） |

**数据流：** 输入 → `SwinTransformerBlock(c1 → 2c)` → `chunk(2)` 均分 → `Bottleneck` 增强一路 → `concat` → `1×1 Conv` 输出 `c2`。

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

YAML 中的出现位置（`HybriDet.yaml` Backbone）：

```yaml
- [-1, 3, SwinBottle, [128, True]]
- [-1, 6, SwinBottle, [256, True]]
- [-1, 6, SwinBottle, [512, True]]
- [-1, 3, SwinBottle, [1024, True]]
```

阅读顺序建议：`SwinBottle` → `SwinTransformerBlock` → `SwinTransformerLayer` / `WindowAttention`。

### 2. CS Attention（Neck 坐标–空间双重注意力）

| 项目 | 位置 |
| --- | --- |
| 类定义 | `block.py` → `CS_Attention`（约 L422–447） |
| 通道注意力 | 同文件 `CoordAtt`（约 L384–419），内部用 `h_swish` / `h_sigmoid` |
| 空间注意力 | `CS_Attention.spatial_attention`：两层 `7×7 Conv` + BN + ReLU |
| YAML 用法 | Neck：`[-1, 3, CS_Attention, [c]]` |

**数据流：** `CoordAtt`（H/W 方向池化 → 通道注意力）→ BN → 与输入相乘 → `spatial_attention` → sigmoid → 再与特征相乘。

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

YAML 中的出现位置（`HybriDet.yaml` Neck）：

```yaml
- [-1, 3, CS_Attention, [512]]
- [-1, 3, CS_Attention, [256]]
- [-1, 3, CS_Attention, [512]]
- [-1, 3, CS_Attention, [1024]]
```

阅读顺序建议：`CS_Attention` → `CoordAtt` → `h_swish` / `h_sigmoid`。

---

## 常见问题澄清（FAQ）

### 1. SwinBottle 中的通道分割比例

**问题。** 论文第 3.2 节写特征图被分割成“三个维度相同的特征张量”，但表 1 中通道数 `[32, 64, 128, 256]` 不能被 3 整除。参数量估算显示 `[C/2, C/4, C/4]` 比 `[C/3, C/3, C/3]` 更接近表 1。实现里实际采用哪种分割？

**回答。** 开源代码**既不是** `[C/3, C/3, C/3]`，**也不是** `[C/2, C/4, C/4]`。

`SwinBottle` 采用与 **C2f 一致** 的结构（见 `ultralytics/nn/modules/block.py`）：

1. 扩展比 `e = 0.5`，隐层宽度 `c = C / 2`（`C` 为该 stage 输出通道，如 32 / 64 / 128 / 256）。
2. Swin Transformer 将特征映射到 `2c = C` 通道。
3. 对这 `C` 个通道用 `chunk(2, 1)` **均分为两路**，每路宽度为 `C/2`。
4. 第三条路径由作用于第二路的 `Bottleneck` 生成，宽度同样为 `C/2`。
5. 因此在 `n = 1` 时，三路特征张量形状相同，均为 **`[C/2, C/2, C/2]`**，再经 `1×1` 卷积融合输出。

论文中“三个维度相同的特征张量”指 expand–split 之后宽度为 **`C/2`** 的三路分支，**不是**把原始 `C` 直接三等分为 `C/3`。

```python
# SwinBottle 前向（简化）
self.c = int(c2 * 0.5)                          # C/2
self.transformer = SwinTransformerBlock(..., 2 * self.c, ...)
y = list(self.transformer(x).chunk(2, 1))      # [C/2, C/2]
y.extend(m(y[-1]) for m in self.m)             # + Bottleneck → C/2
return self.cv2(torch.cat(y, 1))
```

---

## 引用

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

## 致谢

本项目基于 [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) 开发。使用上游框架请引用 Ultralytics；使用本方法请引用 HybriDet（Dong & Wang, *Remote Sensing* 2025）。感谢 Ultralytics 团队与 FASDD 数据集作者。

软件引用元数据见 [`CITATION.cff`](CITATION.cff)。

## 许可证

详见 [LICENSE](LICENSE)。源自 Ultralytics 的代码遵循 AGPL-3.0（另有说明除外）。
