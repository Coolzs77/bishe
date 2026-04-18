# 基于红外图像的行人多目标检测与跟踪系统

## 项目定位

- 检测/跟踪结果汇总与可视化

已移除部署、RKNN 转换和嵌入式相关目录，避免与当前实验主线混杂。

## 当前目录

```text
bishe/
├── configs/                 配置文件
├── data/                    数据与视频
├── docs/                    当前文档
├── model/                   检测与跟踪模型相关资源
├── outputs/                 训练与评估输出
├── scripts/                 数据、训练、评估脚本
├── yolov5/                  本地 YOLOv5 代码
└── 项目说明.md              项目概述
```

### 1. 训练

```bash
python scripts/train/train_yolov5.py --config configs/train_config.yaml
```

说明：该脚本现在是纯训练入口，不再负责自动克隆 YOLOv5 或自动下载权重；训练前需保证 yolov5 目录与 yolov5/yolov5s.pt 已存在。

### 2. 消融实验

```bash
python scripts/train/train_ablation.py --profile controlled
```

如需跑“各实验各自尽量调优”的口径：

```bash
python scripts/train/train_ablation.py --profile optimal
```

### 3. 检测评估

```bash
python scripts/evaluate/eval_detection.py --config configs/eval_detection.yaml --weights outputs/ablation_study/ablation_exp07_eiou/weights/best.pt
```

说明：检测评估默认会在 outputs/detection 下为每次运行创建独立批次目录，目录内保存 summary.json；批量评估默认还会保存 summary.csv。metric 评估不再额外生成独立的 val_detection 根目录。

### 4. 跟踪评估

```bash
python scripts/evaluate/eval_tracking.py --config configs/tracking_config.yaml --weights outputs/ablation_study/ablation_exp07_eiou/weights/best.pt --tracker deepsort --output outputs/tracking/current --half --no-save-vid --no-save-txt --no-overlay
```

### 5. 跟踪结果绘图

```bash
python scripts/evaluate/plot_tracking_algorithm_comparison.py --config configs/plot_tracking_comparison.yaml
```

### 6. RKNN 模型转换（Ubuntu 端）

以下命令在 Ubuntu（已安装 rknn-toolkit2）上执行。

#### 6.1 普通量化（normal）

```bash
cd ~/bishe/deploy/rv1126b_yolov5/python

# Baseline
python convert_yolov5_to_rknn.py \
  --onnx ../model/best_baseline.onnx \
  --dataset ../calibration_dataset.txt \
  --val-dir ~/bishe/val \
  --output ../model/best_baseline.rknn \
  --target rv1126b --quant i8 --quant-algo normal

# EIoU
python convert_yolov5_to_rknn.py \
  --onnx ../model/best_eiou.onnx \
  --dataset ../calibration_dataset.txt \
  --val-dir ~/bishe/val \
  --output ../model/best_eiou.rknn \
  --target rv1126b --quant i8 --quant-algo normal

# Ghost+EIoU
python convert_yolov5_to_rknn.py \
  --onnx ../model/best_ghost_eiou.onnx \
  --dataset ../calibration_dataset.txt \
  --val-dir ~/bishe/val \
  --output ../model/best_ghost_eiou.rknn \
  --target rv1126b --quant i8 --quant-algo normal
```

#### 6.2 KL 散度量化（推荐，适配红外小目标）

KL 散度量化通过最小化量化前后激活值分布的 KL 散度来确定量化截断点，相比 min-max（normal）量化能更好地保留尾部激活精度，特别适合红外小目标场景。转换速度快，内存占用合理。

```bash
cd ~/bishe/deploy/rv1126b_yolov5/python

# EIoU (kl)
python convert_yolov5_to_rknn.py \
  --onnx ../model/best_eiou.onnx \
  --dataset ../calibration_dataset.txt \
  --val-dir ~/bishe/val \
  --output ../model/best_eiou_kl.rknn \
  --target rv1126b --quant i8 --opt-level 3 --quant-algo kl_divergence

# Baseline (kl)
python convert_yolov5_to_rknn.py \
  --onnx ../model/best_baseline.onnx \
  --dataset ../calibration_dataset.txt \
  --val-dir ~/bishe/val \
  --output ../model/best_baseline_kl.rknn \
  --target rv1126b --quant i8 --opt-level 3 --quant-algo kl_divergence

# Ghost+EIoU (kl)
python convert_yolov5_to_rknn.py \
  --onnx ../model/best_ghost_eiou.onnx \
  --dataset ../calibration_dataset.txt \
  --val-dir ~/bishe/val \
  --output ../model/best_ghost_eiou_kl.rknn \
  --target rv1126b --quant i8 --opt-level 3 --quant-algo kl_divergence
```

#### 6.3 转换参数说明

| 参数 | 含义 |
|------|------|
| `--onnx` | 输入 ONNX 模型路径 |
| `--dataset` | 红外校准图片列表文件（每行一张图片路径） |
| `--val-dir` | Ubuntu 上校准图片实际所在目录（自动修正 Windows 路径） |
| `--output` | 输出 RKNN 模型路径 |
| `--target` | 目标芯片，固定 `rv1126b` |
| `--quant` | `i8` = INT8 量化（推荐），`fp` = 浮点 |
| `--opt-level` | RKNN 优化等级 0\~3（默认 3，最大优化） |
| `--quant-algo` | `normal` = min-max 默认量化，`kl_divergence` = KL 散度量化（推荐） |

## 当前保留的评估脚本

- scripts/evaluate/eval_detection.py: 检测指标与批量消融评估
- scripts/evaluate/eval_tracking.py: 单跟踪器批量视频评估
- scripts/evaluate/plot_eval_summary.py: 检测消融结果绘图
- scripts/evaluate/plot_tracking_algorithm_comparison.py: 跟踪结果汇总绘图

## 当前保留的数据工具

- scripts/data/prepare_flir.py: 整理 FLIR 数据集为训练/验证目录
- scripts/data/images_to_video.py: 将 FLIR 连续图像帧整理为热视频文件

## 当前建议流程

1. 先完成单变量检测消融。
2. 用 detection_eval_batch 的结果筛检测模型。
3. 仅对入选检测模型跑 eval_tracking。
4. 用 plot_tracking_algorithm_comparison.py 汇总跟踪结果。

训练配置边界：

- configs/train_config.yaml 只给 scripts/train/train_yolov5.py 使用。
- configs/ablation/train_profile_controlled.yaml 与 configs/ablation/train_profile_optimal.yaml 只给 scripts/train/train_ablation.py 使用。

评估配置边界：

- configs/eval_detection.yaml 只给 scripts/evaluate/eval_detection.py 使用。
- configs/tracking_config.yaml 只给 scripts/evaluate/eval_tracking.py 使用。
- 评估脚本中的命令行参数只作为临时覆盖，不再承担默认参数来源。

绘图配置边界：

- configs/plot_eval_summary.yaml 只给 scripts/evaluate/plot_eval_summary.py 使用。
- configs/plot_tracking_comparison.yaml 只给 scripts/evaluate/plot_tracking_algorithm_comparison.py 使用。
- 绘图脚本中的命令行参数只作为临时覆盖。

检测结果绘图：

```bash
python scripts/evaluate/plot_eval_summary.py --config configs/plot_eval_summary.yaml --input-dir outputs/detection/detection_eval_batch_xxx
```

## 说明

- 数据集配置目前只有 train 和 val，没有独立 test。
- 因此当前检测指标属于验证集结果，不应表述为最终独立测试集结论。
- 当前仓库不再包含部署链路；如果后续恢复，需要单独建立分支重建。
