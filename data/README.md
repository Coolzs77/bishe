# 数据目录说明

本目录用于存放FLIR数据集的原始数据和处理后的数据。

## 目录结构

```
data/
├── raw/                    # 原始数据目录
│   └── flir/              # FLIR数据集（解压到这里）
│       ├── images_thermal_train/
│       │   ├── coco.json  # 训练集标注文件
│       │   └── data/      # 训练图像
│       └── images_thermal_val/
│           ├── coco.json  # 验证集标注文件
│           └── data/      # 验证图像
│
└── processed/             # 处理后的数据目录
    └── flir/              # YOLO格式数据（自动生成）
        ├── images/
        │   ├── train/     # 训练图像
        │   └── val/       # 验证图像
        ├── labels/
        │   ├── train/     # 训练标签（YOLO格式）
        │   └── val/       # 验证标签（YOLO格式）
        ├── calibration/   # 校准数据（用于模型量化）
        └── dataset_statistics.json  # 数据集统计信息
```

## 使用步骤

### 1. 解压FLIR数据集

将下载的 `FLIR_ADAS_v2.zip` 解压到 `data/raw/flir/` 目录：

```bash
# 在项目根目录执行
unzip FLIR_ADAS_v2.zip -d data/raw/flir/
```

### 2. 处理数据集

运行数据处理脚本：

```bash
python scripts/data/prepare_flir.py --input data/raw/flir --output data/processed/flir
```

处理完成后，`data/processed/flir/` 目录将包含YOLO格式的训练数据。

### 3. 验证数据

检查处理后的数据：

```bash
# 查看统计信息
cat data/processed/flir/dataset_statistics.json

# 查看目录内容
ls data/processed/flir/images/train/ | wc -l    # 训练图像数量
ls data/processed/flir/labels/train/ | wc -l    # 训练标签数量
```

## 注意事项

- ⚠️ 此目录中的数据文件不会提交到Git仓库（已在.gitignore中配置）
- ⚠️ 原始数据集（raw/）大小约为1-2GB
- ⚠️ 处理后的数据（processed/）大小取决于设置的图像尺寸
- ✅ 确保有足够的磁盘空间（建议至少5GB）

## 数据集信息

### FLIR ADAS v2 数据集

- **来源**: FLIR Systems, Inc.
- **类型**: 热红外图像
- **分辨率**: 640x512 (原始)
- **格式**: JPEG (8-bit)
- **标注**: COCO JSON格式
- **类别**: person, car, bicycle (本项目使用的3个核心类别)
- **场景**: 日间/夜间道路场景

### 数据集统计（典型）

- 训练集: ~9,000 图像
- 验证集: ~1,000 图像
- 总标注框: ~100,000+
- 平均每张图像标注数: ~10

## 故障排除

### 问题1: 解压后找不到标注文件

**现象**: `coco.json` 文件不存在

**解决方案**:
- 检查FLIR数据集的版本和结构
- 标注文件可能在其他位置，如：
  - `annotations/instances_thermal_train.json`
  - `thermal_train/coco.json`
- 查看 `prepare_flir.py` 脚本中的 `possible_paths` 变量

### 问题2: 磁盘空间不足

**现象**: 处理过程中出现写入错误

**解决方案**:
- 清理不需要的文件
- 或将数据目录移动到更大的磁盘
- 使用符号链接：`ln -s /path/to/larger/disk/data ./data`

### 问题3: 处理速度慢

**现象**: 数据处理耗时过长

**解决方案**:
- 减少图像尺寸：`--img-size 512`
- 使用SSD而不是HDD
- 确保有足够的内存

## 更多信息

- 数据处理详细指南: [DATA_PROCESSING_GUIDE.md](../DATA_PROCESSING_GUIDE.md)
- 快速开始: [QUICKSTART.md](../QUICKSTART.md)
- 项目说明: [README.md](../README.md)
