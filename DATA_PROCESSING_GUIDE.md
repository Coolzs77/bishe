# FLIR数据集处理指南

本指南说明如何处理FLIR v2数据集用于训练和验证。

## 前提条件

- 已下载 FLIR ADAS v2 数据集ZIP文件
- Python 3.8+ 环境
- 已安装项目依赖：`pip install -r requirements.txt`

## 数据处理步骤

### 1. 解压数据集

将下载的FLIR数据集ZIP文件解压到 `data/raw/flir/` 目录：

```bash
# 假设ZIP文件名为 FLIR_ADAS_v2.zip
unzip FLIR_ADAS_v2.zip -d data/raw/flir/
```

解压后的目录结构应该类似：

```
data/raw/flir/
├── images_thermal_train/
│   └── data/
│       ├── FLIR_00001.jpeg
│       ├── FLIR_00002.jpeg
│       └── ...
├── images_thermal_val/
│   └── data/
│       ├── FLIR_00001.jpeg
│       └── ...
└── video_thermal_train/ (可选)
```

或者：

```
data/raw/flir/
├── train/
│   └── thermal_8_bit/
│       ├── FLIR_00001.jpeg
│       └── ...
├── val/
│   └── thermal_8_bit/
│       └── ...
└── video/ (可选)
```

### 2. 运行数据处理脚本

使用 `prepare_flir.py` 脚本将FLIR数据集转换为YOLO训练格式：

```bash
python scripts/data/prepare_flir.py --input data/raw/flir --output data/processed/flir
```

#### 可选参数：

- `--split-ratio`: 训练集比例（默认0.8）
- `--img-size`: 目标图像尺寸（默认640）
- `--classes`: 检测类别，逗号分隔（默认：person,car,bicycle）
- `--visualize`: 可视化部分结果
- `--seed`: 随机种子（默认42）

#### 示例：

```bash
# 基本使用
python scripts/data/prepare_flir.py --input data/raw/flir

# 自定义划分比例和图像大小
python scripts/data/prepare_flir.py \
    --input data/raw/flir \
    --output data/processed/flir \
    --split-ratio 0.85 \
    --img-size 640

# 可视化结果
python scripts/data/prepare_flir.py \
    --input data/raw/flir \
    --visualize
```

### 3. 验证处理结果

处理完成后，检查输出目录结构：

```
data/processed/flir/
├── images/
│   ├── train/          # 训练图像
│   │   ├── FLIR_00001.jpg
│   │   └── ...
│   └── val/            # 验证图像
│       └── ...
├── labels/
│   ├── train/          # 训练标签（YOLO格式）
│   │   ├── FLIR_00001.txt
│   │   └── ...
│   └── val/            # 验证标签
│       └── ...
└── calibration/        # 量化校准数据（用于模型部署）
    └── ...
```

### 4. 检查数据集统计信息

脚本会在输出目录生成 `dataset_statistics.json` 文件，包含：

- 总图像数
- 训练/验证图像数
- 各类别实例数
- 图像尺寸分布
- 目标大小分布

查看统计信息：

```bash
cat data/processed/flir/dataset_statistics.json
```

## FLIR数据集类别说明

FLIR ADAS v2 数据集包含以下类别，本项目使用其中3个核心类别：

### 使用的类别（3类）

| 类别 | 索引 | 说明 |
|------|------|------|
| person | 0 | 行人 |
| car | 1 | 汽车 |
| bicycle | 2 | 自行车 |

### 原始数据集其他类别

原始FLIR数据集还包含：motor, bus, train, truck, light, hydrant, sign, dog, skateboard, stroller, scooter 等类别，这些类别在处理时会被过滤掉。

如需使用其他类别，可以修改 `configs/dataset.yaml` 中的 `nc` 和 `names` 配置。

## 常见问题

### Q1: 找不到标注文件

**问题**: `FileNotFoundError: 未找到train标注文件`

**解决**: 
- 检查解压后的目录结构
- 确保包含 `coco.json` 或 `instances_thermal_train.json` 标注文件
- 如果标注文件在其他位置，修改 `prepare_flir.py` 中的 `possible_paths`

### Q2: 图像文件找不到

**问题**: 处理过程中跳过大量图像

**解决**:
- 确认图像文件路径与标注文件中的路径一致
- FLIR数据集图像可能在 `data` 子目录或 `thermal_8_bit` 子目录中
- 检查 `prepare_flir.py` 中的 `find_image_file()` 函数

### Q3: 内存不足

**问题**: 处理大量图像时内存溢出

**解决**:
- 减少 `--img-size` 参数值
- 分批处理数据集
- 增加系统交换空间

## 下一步

数据处理完成后，可以：

1. **训练模型**:
   ```bash
   python scripts/train/train_yolov5.py --config configs/train_config.yaml
   ```

2. **评估模型**:
   ```bash
   python scripts/evaluate/eval_detection.py --weights outputs/weights/best.pt
   ```

3. **运行完整流程**:
   ```bash
   python main.py --mode full
   ```

## 参考资料

- FLIR官方数据集: https://www.flir.com/oem/adas/adas-dataset-form/
- YOLOv5文档: https://github.com/ultralytics/yolov5
- 项目README: README.md
