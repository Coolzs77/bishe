# 快速开始指南

本指南帮助你快速搭建基于FLIR红外数据集的目标检测与跟踪系统。

## 前提条件

- Python 3.8+
- CUDA 10.2+ (可选，用于GPU加速)
- 已下载FLIR ADAS v2数据集ZIP文件

## 快速开始（5步完成）

### 步骤 1: 克隆仓库

```bash
git clone https://github.com/Coolzs77/bishe.git
cd bishe
```

### 步骤 2: 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖包
pip install -r requirements.txt
```

### 步骤 3: 准备数据集

```bash
# 3.1 创建数据目录
mkdir -p data/raw/flir

# 3.2 解压FLIR数据集到data/raw/flir/
# 假设你的ZIP文件在当前目录
unzip FLIR_ADAS_v2.zip -d data/raw/flir/

# 3.3 处理数据集（转换为YOLO格式）
python scripts/data/prepare_flir.py --input data/raw/flir --output data/processed/flir

# 这将创建以下结构：
# data/processed/flir/
#   ├── images/train/    # 训练图像
#   ├── images/val/      # 验证图像
#   ├── labels/train/    # 训练标签
#   └── labels/val/      # 验证标签
```

### 步骤 4: 训练模型

```bash
# 使用默认配置训练YOLOv5s模型
python scripts/train/train_yolov5.py --config configs/train_config.yaml

# 训练过程会保存到 outputs/weights/
# 最佳模型: outputs/weights/best.pt
# 最后一个epoch: outputs/weights/last.pt
```

### 步骤 5: 评估与测试

```bash
# 评估检测性能
python scripts/evaluate/eval_detection.py --weights outputs/weights/best.pt

# 在验证集上运行推理
python main.py --mode detect --weights outputs/weights/best.pt
```

## 目录结构

```
bishe/
├── configs/              # 配置文件
│   ├── dataset.yaml     # 数据集配置（类别、路径）
│   ├── train_config.yaml # 训练配置
│   └── ...
├── data/                # 数据目录（不提交到git）
│   ├── raw/flir/        # 原始FLIR数据
│   └── processed/flir/  # 处理后的YOLO格式数据
├── models/              # 模型定义
│   └── yolov5/          # YOLOv5相关模块
├── scripts/             # 各类脚本
│   ├── data/            # 数据处理脚本
│   ├── train/           # 训练脚本
│   ├── evaluate/        # 评估脚本
│   └── deploy/          # 部署脚本
├── src/                 # 核心源代码
│   ├── detection/       # 检测模块
│   ├── tracking/        # 跟踪模块
│   ├── utils/           # 工具函数
│   └── ...
├── outputs/             # 输出目录（不提交到git）
│   ├── weights/         # 训练权重
│   ├── logs/            # 训练日志
│   └── results/         # 评估结果
└── main.py              # 主程序入口
```

## 数据集说明

### FLIR ADAS v2 数据集

- **类型**: 热红外图像
- **分辨率**: 640x512
- **类别**: 本项目使用3个核心类别
  - person (行人)
  - car (汽车)
  - bicycle (自行车)
- **数据量**: 
  - 训练集: ~10,000+ 图像
  - 验证集: ~1,000+ 图像

## 配置文件说明

### configs/dataset.yaml

配置数据集路径和类别：

```yaml
path: data/processed/flir
train: images/train
val: images/val
nc: 3
names: ['person', 'car', 'bicycle']
```

### configs/train_config.yaml

配置训练参数：

```yaml
epochs: 100
batch_size: 32
img_size: 640
learning_rate: 0.01
```

## 常用命令

### 数据处理

```bash
# 基本处理
python scripts/data/prepare_flir.py --input data/raw/flir

# 自定义参数
python scripts/data/prepare_flir.py \
    --input data/raw/flir \
    --output data/processed/flir \
    --split-ratio 0.85 \
    --img-size 640 \
    --visualize
```

### 模型训练

```bash
# 从头开始训练
python scripts/train/train_yolov5.py --config configs/train_config.yaml

# 从预训练模型开始
python scripts/train/train_yolov5.py \
    --config configs/train_config.yaml \
    --weights yolov5s.pt

# 继续训练
python scripts/train/train_yolov5.py \
    --config configs/train_config.yaml \
    --weights outputs/weights/last.pt \
    --resume
```

### 模型评估

```bash
# 评估检测性能
python scripts/evaluate/eval_detection.py \
    --weights outputs/weights/best.pt \
    --data configs/dataset.yaml

# 可视化结果
python scripts/evaluate/eval_detection.py \
    --weights outputs/weights/best.pt \
    --save-txt \
    --save-conf
```

### 模型部署

```bash
# 导出ONNX模型
python scripts/deploy/export_model.py \
    --weights outputs/weights/best.pt \
    --format onnx

# 转换为RKNN（用于RV1126）
python scripts/deploy/convert_to_rknn.py \
    --onnx outputs/weights/best.onnx \
    --output models/rknn/best.rknn
```

## 常见问题

### Q1: 数据处理时找不到标注文件

**A**: 确保FLIR数据集完整解压，标注文件通常在以下位置：
- `images_thermal_train/coco.json`
- `images_thermal_val/coco.json`

### Q2: 训练时GPU内存不足

**A**: 减小批次大小：
```bash
# 在train_config.yaml中修改
batch_size: 16  # 或更小
```

### Q3: 如何使用预训练权重

**A**: 下载YOLOv5预训练权重：
```bash
wget https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt
```

然后在训练时指定：
```bash
python scripts/train/train_yolov5.py \
    --config configs/train_config.yaml \
    --weights yolov5s.pt
```

## 下一步

- 查看 [README.md](README.md) 了解详细项目结构
- 查看 [DATA_PROCESSING_GUIDE.md](DATA_PROCESSING_GUIDE.md) 了解数据处理细节
- 查看 `docs/` 目录获取更多文档

## 获取帮助

- 提交Issue: https://github.com/Coolzs77/bishe/issues
- 查看文档: `docs/` 目录
- 参考YOLOv5文档: https://github.com/ultralytics/yolov5

---

**祝你训练顺利！** 🚀
