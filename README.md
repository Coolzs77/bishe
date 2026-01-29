# 基于红外图像的行人多目标检测与跟踪系统

## 项目概述

本项目实现了一个基于红外图像的行人多目标检测与跟踪系统，主要功能包括：
- 使用改进的YOLOv5进行红外图像目标检测
- 实现并对比多种跟踪算法（DeepSORT、ByteTrack、CenterTrack）
- 将模型部署到RV1126嵌入式平台

---

## 项目结构

```
bishe/
├── README.md                    # 本文件 - 项目总体说明
├── 项目说明.md                  # 项目概要说明
├── requirements.txt             # Python依赖包列表
├── .gitignore                   # Git忽略文件配置
├── main.py                      # 主函数 - 涵盖全流程
│
├── configs/                     # 配置文件目录
├── data/                        # 数据目录
├── models/                      # 模型定义目录
├── src/                         # 源代码目录
├── scripts/                     # 脚本目录
├── embedded/                    # 嵌入式代码目录
├── docs/                        # 文档目录
├── tests/                       # 测试目录
└── outputs/                     # 输出目录
```

---

## 目录详细说明

### 📁 configs/ - 配置文件目录

存放所有配置文件，统一管理项目参数。

| 文件 | 用途 | 使用方式 |
|------|------|----------|
| `dataset.yaml` | 数据集配置（路径、类别、划分比例等） | 被数据处理和训练脚本读取 |
| `train_config.yaml` | 训练配置（学习率、批次大小、轮次等） | 被训练脚本读取 |
| `tracking_config.yaml` | 跟踪算法配置（各跟踪器参数） | 被跟踪模块读取 |
| `deploy_config.yaml` | 部署配置（量化参数、目标平台等） | 被部署脚本读取 |

**使用示例：**
```python
import yaml
with open('configs/train_config.yaml', 'r') as f:
    config = yaml.safe_load(f)
```

---

### 📁 data/ - 数据目录

存放原始数据、处理后数据和标注文件。

| 子目录 | 用途 |
|--------|------|
| `raw/` | 原始数据集（FLIR、KAIST红外数据集） |
| `processed/` | 处理后的数据（统一格式、划分后的训练/验证/测试集） |
| `annotations/` | 标注文件（YOLO格式的txt标注） |

**注意：** 此目录不纳入版本控制，需要运行数据下载脚本生成。

---

### 📁 models/ - 模型定义目录

存放模型配置、自定义模块和训练权重。

#### models/yolov5/ - YOLOv5相关模型

| 文件/目录 | 用途 | 说明 |
|-----------|------|------|
| `yolov5s_infrared.yaml` | 红外优化的YOLOv5s配置 | 定义网络结构，包含针对红外图像的优化 |
| `modules/` | 自定义模块 | 注意力机制（SE、CBAM、CA）等 |
| `modules/attention.py` | 注意力机制实现 | SEAttention、CBAM、CoordAttention等 |
| `backbone/` | 轻量化骨干网络 | 适合嵌入式部署的轻量化设计 |
| `backbone/lightweight.py` | 轻量化骨干实现 | GhostConv、ShuffleBlock等 |

#### models/tracking/ - 跟踪算法配置

| 目录 | 用途 | 说明 |
|------|------|------|
| `deepsort/` | DeepSORT配置 | 包含config.yaml配置文件 |
| `bytetrack/` | ByteTrack配置 | 包含config.yaml配置文件 |
| `centertrack/` | CenterTrack配置 | 包含config.yaml配置文件 |

#### models/rknn/ - RKNN量化模型

| 文件 | 用途 |
|------|------|
| `README.md` | RKNN模型说明 |
| `*.rknn` | 量化后的RKNN模型文件（运行转换脚本后生成） |

---

### 📁 src/ - 源代码目录

核心源代码，包含检测、跟踪、部署、评估等模块。

#### src/detection/ - 目标检测模块

| 文件 | 用途 | 主要类/函数 |
|------|------|-------------|
| `__init__.py` | 模块初始化 | 导出主要类 |
| `detector.py` | 检测器基类 | `BaseDetector`、`DetectionResult` |
| `yolov5_detector.py` | YOLOv5检测器实现 | `YOLOv5Detector` |
| `data_augment.py` | 红外数据增强 | `InfraredDataAugmentor` |

**使用示例：**
```python
from src.detection import YOLOv5Detector

detector = YOLOv5Detector(weights='outputs/weights/best.pt')
results = detector.detect(image)
```

#### src/tracking/ - 多目标跟踪模块

| 文件 | 用途 | 主要类/函数 |
|------|------|-------------|
| `__init__.py` | 模块初始化 | 导出主要类 |
| `tracker.py` | 跟踪器基类 | `BaseTracker`、`TrackObject`、`TrackingResult` |
| `kalman_filter.py` | 卡尔曼滤波器 | `KalmanFilter`、`KalmanBoxTracker` |
| `deepsort_tracker.py` | DeepSORT跟踪器 | `DeepSORTTracker` |
| `bytetrack_tracker.py` | ByteTrack跟踪器 | `ByteTracker` |
| `centertrack_tracker.py` | CenterTrack跟踪器 | `CenterTracker` |

**使用示例：**
```python
from src.tracking import DeepSORTTracker, ByteTracker

tracker = DeepSORTTracker(config_path='configs/tracking_config.yaml')
tracks = tracker.update(detections, frame)
```

#### src/deploy/ - 部署相关模块

| 文件 | 用途 | 主要类/函数 |
|------|------|-------------|
| `__init__.py` | 模块初始化 | 导出主要类 |
| `export_onnx.py` | 导出ONNX模型 | `ONNXExporter` |
| `convert_rknn.py` | 转换RKNN模型 | `RKNNConverter` |
| `quantize.py` | INT8量化 | `QuantizationCalibrator`、`ModelQuantizer` |

**使用示例：**
```python
from src.deploy import ONNXExporter, RKNNConverter

# 导出ONNX
exporter = ONNXExporter(model_path='outputs/weights/best.pt')
exporter.export('outputs/weights/best.onnx')

# 转换RKNN
converter = RKNNConverter(onnx_path='outputs/weights/best.onnx')
converter.convert('outputs/weights/best.rknn')
```

#### src/evaluation/ - 评估模块

| 文件 | 用途 | 主要类/函数 |
|------|------|-------------|
| `__init__.py` | 模块初始化 | 导出主要类 |
| `detection_eval.py` | 检测评估 | `DetectionEvaluator` |
| `tracking_eval.py` | 跟踪评估 | `MOTEvaluator` |

**使用示例：**
```python
from src.evaluation import DetectionEvaluator, MOTEvaluator

# 检测评估
det_eval = DetectionEvaluator()
det_eval.evaluate(predictions, ground_truth)
print(det_eval.get_metrics())  # mAP, Precision, Recall等

# 跟踪评估
mot_eval = MOTEvaluator()
mot_eval.evaluate(tracks, ground_truth)
print(mot_eval.get_metrics())  # MOTA, MOTP, IDF1等
```

#### src/utils/ - 工具函数

| 文件 | 用途 | 主要类/函数 |
|------|------|-------------|
| `__init__.py` | 模块初始化 | 导出主要类 |
| `metrics.py` | 评估指标计算 | `calculate_iou`、`calculate_map`、`calculate_mota` |
| `visualization.py` | 可视化工具 | `Visualizer`、`draw_boxes`、`draw_tracks` |
| `logger.py` | 日志工具 | `LogManager`、`TrainingLogger`、`ProgressBar` |

**使用示例：**
```python
from src.utils import Visualizer, LogManager

# 可视化
vis = Visualizer()
vis.draw_tracks(frame, tracks)
vis.save('output.jpg')

# 日志
logger = LogManager.get_logger('train')
logger.info('Training started')
```

---

### 📁 scripts/ - 脚本目录

可执行脚本，用于数据处理、训练、评估和部署。

#### scripts/data/ - 数据处理脚本

| 文件 | 用途 | 命令行使用 |
|------|------|------------|
| `download_dataset.py` | 自动下载数据集（从公开镜像源） | `python scripts/data/download_dataset.py [--flir] [--kaist]` |
| `prepare_flir.py` | 准备FLIR数据集（转换为YOLO格式） | `python scripts/data/prepare_flir.py --input data/raw/flir --output data/processed` |
| `prepare_kaist.py` | 准备KAIST数据集（转换为YOLO格式） | `python scripts/data/prepare_kaist.py --input data/raw/kaist --output data/processed` |

**数据下载说明：**
- `download_dataset.py` 会尝试从多个公开镜像源自动下载数据集
- 由于数据集许可限制，自动下载通常需要手动辅助
- 脚本提供详细的下载指引，并支持手动下载后继续处理
- 建议直接访问官方网站获取数据集

#### scripts/train/ - 训练脚本

| 文件 | 用途 | 命令行使用 |
|------|------|------------|
| `train_yolov5.py` | 训练YOLOv5 | `python scripts/train/train_yolov5.py --config configs/train_config.yaml` |
| `ablation_study.py` | 消融实验 | `python scripts/train/ablation_study.py --config configs/train_config.yaml` |

#### scripts/evaluate/ - 评估脚本

| 文件 | 用途 | 命令行使用 |
|------|------|------------|
| `eval_detection.py` | 评估检测性能 | `python scripts/evaluate/eval_detection.py --weights outputs/weights/best.pt --data data/processed/test` |
| `eval_tracking.py` | 评估跟踪性能 | `python scripts/evaluate/eval_tracking.py --tracker deepsort --data data/processed/test` |
| `compare_trackers.py` | 对比跟踪算法 | `python scripts/evaluate/compare_trackers.py --output outputs/results/comparison.json` |

#### scripts/deploy/ - 部署脚本

| 文件 | 用途 | 命令行使用 |
|------|------|------------|
| `export_model.py` | 导出模型 | `python scripts/deploy/export_model.py --weights outputs/weights/best.pt --format onnx` |
| `convert_to_rknn.py` | 转换为RKNN | `python scripts/deploy/convert_to_rknn.py --onnx outputs/weights/best.onnx --output models/rknn/best.rknn` |
| `test_rknn.py` | 测试RKNN模型 | `python scripts/deploy/test_rknn.py --model models/rknn/best.rknn --image test.jpg` |

---

### 📁 embedded/ - 嵌入式代码目录

RV1126嵌入式平台部署代码。

| 文件/目录 | 用途 |
|-----------|------|
| `CMakeLists.txt` | CMake构建配置 |
| `toolchain.cmake` | 交叉编译工具链配置 |
| `部署说明.md` | 嵌入式部署详细说明 |

#### embedded/include/ - 头文件

| 文件 | 用途 |
|------|------|
| `detector.h` | 检测器头文件 |
| `tracker.h` | 跟踪器头文件 |
| `pipeline.h` | 处理流水线头文件 |

#### embedded/src/ - 源文件

| 文件 | 用途 |
|------|------|
| `main.cpp` | 主程序入口 |
| `detector.cpp` | 检测器实现 |
| `tracker.cpp` | 跟踪器实现 |
| `pipeline.cpp` | 处理流水线实现 |

#### embedded/configs/ - 嵌入式配置

存放嵌入式平台的配置文件。

#### embedded/lib/ - 依赖库

存放RKNN等依赖库文件。

**编译命令：**
```bash
cd embedded
mkdir build && cd build
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchain.cmake ..
make -j4
```

---

### 📁 docs/ - 文档目录

| 文件 | 用途 |
|------|------|
| `完整实施指南.md` | 完整的项目实施指南 |
| `脚本执行文档.md` | 各脚本的详细执行说明 |
| `部署指南.md` | 嵌入式部署指南 |

---

### 📁 tests/ - 测试目录

| 文件 | 用途 | 运行命令 |
|------|------|----------|
| `test_detection.py` | 检测模块测试 | `pytest tests/test_detection.py` |
| `test_tracking.py` | 跟踪模块测试 | `pytest tests/test_tracking.py` |
| `test_utils.py` | 工具模块测试 | `pytest tests/test_utils.py` |

**运行所有测试：**
```bash
pytest tests/ -v
```

---

### 📁 outputs/ - 输出目录

存放所有输出文件，不纳入版本控制。

| 子目录 | 用途 |
|--------|------|
| `weights/` | 模型权重文件（.pt、.onnx、.rknn） |
| `logs/` | 训练日志和TensorBoard日志 |
| `results/` | 评估结果（JSON、CSV等） |
| `visualizations/` | 可视化结果图片和视频 |

---

## 快速开始

### 1. 环境安装

```bash
# 克隆项目
git clone https://github.com/Coolzs77/bishe.git
cd bishe

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

```bash
# 自动下载FLIR和KAIST数据集（推荐）
python scripts/data/download_dataset.py

# 仅下载FLIR数据集
python scripts/data/download_dataset.py --flir

# 仅下载KAIST数据集
python scripts/data/download_dataset.py --kaist

# 准备FLIR数据集（转换为YOLO格式）
python scripts/data/prepare_flir.py --input data/raw/flir --output data/processed

# 准备KAIST数据集（转换为YOLO格式）
python scripts/data/prepare_kaist.py --input data/raw/kaist --output data/processed
```

**注意：**
- 脚本会尝试从多个公开镜像源自动下载数据集
- **由于数据集通常需要注册或授权**，自动下载可能会失败
- 失败时脚本会提供详细的手动下载指南：
  - **FLIR数据集**: 访问官方网站注册后下载（https://www.flir.com/oem/adas/adas-dataset-form/）
  - **KAIST数据集**: 从GitHub仓库获取下载链接（https://github.com/SoonminHwang/rgbt-ped-detection）
- 手动下载后，将ZIP文件放到 `data/raw/flir` 或 `data/raw/kaist` 目录即可

### 3. 模型训练

```bash
# 训练YOLOv5检测器
python scripts/train/train_yolov5.py --config configs/train_config.yaml
```

### 4. 评估

```bash
# 评估检测性能
python scripts/evaluate/eval_detection.py --weights outputs/weights/best.pt

# 评估跟踪性能
python scripts/evaluate/eval_tracking.py --tracker deepsort

# 对比跟踪算法
python scripts/evaluate/compare_trackers.py
```

### 5. 部署

```bash
# 导出ONNX模型
python scripts/deploy/export_model.py --weights outputs/weights/best.pt --format onnx

# 转换为RKNN模型
python scripts/deploy/convert_to_rknn.py --onnx outputs/weights/best.onnx
```

### 6. 运行完整流程

```bash
# 使用主函数运行完整流程
python main.py --mode full

# 仅运行特定阶段
python main.py --mode train
python main.py --mode evaluate
python main.py --mode deploy
```

---

## 主要依赖

- Python >= 3.8
- PyTorch >= 1.10
- OpenCV >= 4.5
- NumPy >= 1.20
- RKNN-Toolkit2（用于模型转换）

详见 `requirements.txt`

---

## 许可证

本项目仅用于学术研究和毕业设计。

---

## 联系方式

如有问题，请联系项目作者。
