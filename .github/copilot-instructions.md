# 本仓库 Copilot 协作说明

## 构建、测试与检查命令

### 环境安装（主流程）
```bash
pip install -r requirements.txt
```

### 核心实验流程
```bash
# 1) 处理 FLIR 数据集（生成 data/processed/flir/dataset.yaml）
python scripts/data/prepare_flir.py --input data/raw/flir

# 2) 单模型训练
python scripts/train/train_yolov5.py --config configs/train_config.yaml

# 3) 消融训练
python scripts/train/train_ablation.py --profile controlled

# 4) 检测评估（单模型）
python scripts/evaluate/eval_detection.py --config configs/eval_detection.yaml --weights outputs/ablation_study/ablation_exp07_eiou/weights/best.pt

# 5) 跟踪评估
python scripts/evaluate/eval_tracking.py --config configs/tracking_config.yaml --weights outputs/ablation_study/ablation_exp07_eiou/weights/best.pt --tracker deepsort --output outputs/tracking/current --half --no-save-vid --no-save-txt --no-overlay
```

### 单目标运行（作为“单测等价入口”）
```bash
# 仅运行一个消融实验
python scripts/train/train_ablation.py --profile controlled --only exp7
```

### Lint
仓库根文档与配置中未定义统一 lint 命令，不要臆造新命令。

### 可选部署构建（仅 RV1126B 路径）
```bash
bash deploy/rv1126b_yolov5/build_rv1126b.sh
```

## 高层架构

本仓库采用 **脚本编排层** + **核心算法模块层** 的结构：

1. `scripts/` 是流程入口与编排层：
   - `scripts/data/*`：FLIR 数据预处理、图像序列转视频。
   - `scripts/train/*`：单模型训练（`train_yolov5.py`）与标准化消融训练（`train_ablation.py`）。
   - `scripts/evaluate/*`：检测测速/指标评估、跟踪评估、结果绘图。
2. `src/` 是可复用实现层：
   - `src/detection/`：检测器抽象与 YOLOv5 检测封装（PyTorch/ONNX）。
   - `src/tracking/`：`BaseTracker`、`TrackObject` 及 DeepSORT/ByteTrack/CenterTrack 实现与共享匹配逻辑。
   - `src/evaluation/`：检测与 MOT 评估计算模块。
3. `yolov5/` 是本地保留的上游 YOLOv5 代码，训练和 metric 评估会直接调用（如 `val.py`）。
4. `configs/` 提供按任务划分的 YAML 默认配置；脚本按“配置 + CLI 覆盖”合并参数。
5. `outputs/` 是统一产物目录（`weights`、`ablation_study`、`detection`、`tracking`），评估/跟踪结果通常按时间戳分批次输出。

## 本仓库关键约定

1. **配置边界是硬约束**：
   - `configs/train_config.yaml` 只给 `scripts/train/train_yolov5.py` 用。
   - `configs/ablation/train_profile_*.yaml` 只给 `scripts/train/train_ablation.py` 用。
   - `configs/eval_detection.yaml` 只给 `scripts/evaluate/eval_detection.py` 用。
   - `configs/tracking_config.yaml` 只给 `scripts/evaluate/eval_tracking.py` 用。
   不要跨脚本复用默认参数来源。

2. **CLI 可覆盖配置，但配置仍是默认真值来源**：
   多数脚本遵循 `load_config` + `config_get` + `pick/override` 的合并模式；新增参数时保持这一模式。

3. **默认按项目根目录解析相对路径**：
   脚本普遍通过 `ROOT = Path(__file__).parents[...]` 计算项目根并归一化路径；新增路径参数时保持一致。

4. **消融实验命名与顺序是共享约定**：
   `ablation_exp01_*` 到 `ablation_exp13_*` 同时用于训练、阶段筛选（`stage1`/`stage2`）、权重发现和汇总排序。

5. **当前主线类别固定为两类**：
   数据处理与检测默认类别对齐 `person`、`car`；若调整类别集，需同步修改数据转换、模型假设和评估汇总。

6. **跟踪结果有“单视频产物 + 批量汇总”双层输出**：
   `eval_tracking.py` 会在每个视频目录写 `metrics.json`，并在批量目录写 `summary_metrics.csv/json`；下游绘图脚本依赖该结构。
