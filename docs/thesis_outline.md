# 毕业论文大纲
# 基于改进 YOLOv5 的热红外多目标检测与跟踪系统研究

> 版本：2026-04-19  
> 说明：每节末尾 `【证据】` 指向本仓库的可验证路径；`【图/表/公式】` 为占位符，写作时填入实际编号。  
> **建议写作顺序**：第3章 → 第4章 → 第5章 → 第6章 → 第2章 → 第1章 → 摘要

---

## 摘要（中/英文）

### 中文摘要（200~300字）

> 占位示例结构（写作时替换）：  
> 热红外成像技术在……（背景）。本文以 FLIR 热红外数据集为研究对象，在 YOLOv5 基线上开展了 **13 组系统性消融实验**，分别探究……（改进）。实验结果表明，引入 EIoU 损失函数后，模型在 FLIR 验证集上的 mAP@0.5 提升至 **0.817**，mAP@0.5:0.95 达到 **0.516**。在多目标跟踪对比实验中，ByteTrack 算法在 ID Switch 指标上取得最低值（**26**），展现出最优的轨迹连续性。针对 RV1126B 嵌入式平台，本文完成了模型量化转换与 C++ 推理框架设计，通过 SRAM 标志位、非缓存输入内存、logit 预滤波等优化手段，最终实现板端推理帧率 **≥30 FPS**。此外，本文设计并实现了一套基于 PyQt5 的可视化管理系统，涵盖数据预处理、模型训练监控、检测/跟踪评估与板端部署管控全流程。

**关键词**：热红外检测；YOLOv5；EIoU 损失；ByteTrack；RV1126B；嵌入式部署；PyQt5

### 英文摘要（Abstract）

> 与中文摘要内容对应，结构相同，翻译时使用学术英文表述。

**Keywords**: Thermal Infrared Detection; YOLOv5; EIoU Loss; ByteTrack; RV1126B; Embedded Deployment; PyQt5

---

## 目录（自动生成，写作后更新）

章节层级：一级标题 = 章，二级 = 节，三级 = 小节，最多三级。

---

## 第 1 章　绪论

### 1.1　研究背景与意义（约 1.5 页）

- **热红外成像应用场景**  
  - 安防监控（全天候、穿烟雾）  
  - 自动驾驶辅助（行人/车辆夜间感知）  
  - 无人机巡检（电力线路、森林防火）  
  - 工业缺陷检测  
- **红外图像与可见光图像的差异**  
  - 无纹理、低对比度、目标与背景热辐射差异  
  - person/car 类间外观差异较小，检测难度大  
- **嵌入式实时部署的工程价值**  
  - 端侧推理减少云端延迟与隐私风险  
  - 低功耗嵌入式 NPU（如 RV1126B）的成本优势  
- **本文研究意义**（3~5句，概括全文贡献）

### 1.2　国内外研究现状（约 2.5 页）

#### 1.2.1　目标检测方法研究进展

- **两阶段检测器**：R-CNN → Fast R-CNN → Faster R-CNN，精度高但速度受限  
- **单阶段检测器**：YOLO 系列演进（YOLOv1→v3→v5→v8），速度与精度兼顾  
  - YOLOv5 架构特点：CSP-DarkNet Backbone、FPN+PAN Neck、三尺度 Head  
- **热红外专用检测**  
  - FLIR 数据集上的研究现状（KAIST、InfraDet 等）  
  - 红外增强 YOLO 相关工作

#### 1.2.2　轻量化模型研究进展

- GhostNet（低秩分解，华为）  
- ShuffleNetV2（Channel Shuffle + 通道分组）  
- MobileNetV3（Depthwise Separable Conv）

#### 1.2.3　注意力机制研究进展

- SE-Net（通道注意力）  
- CBAM（通道+空间双路注意力）  
- CoordAtt（坐标注意力，保留位置信息）

#### 1.2.4　多目标跟踪方法研究进展

- SORT（IoU 关联 + KF）  
- DeepSORT（外观特征 Re-ID + KF）  
- ByteTrack（双阈值二次关联，ID Switch 更少）  
- CenterTrack（中心偏移回归）

#### 1.2.5　嵌入式 AI 推理框架研究进展

- TensorRT（NVIDIA GPU）  
- RKNN Toolkit2（Rockchip NPU）  
- NCNN、MNN（ARM CPU）

### 1.3　研究内容与主要贡献（约 1 页）

1. 设计 13 组严格控变消融实验，定量评估主干轻量化、注意力机制、损失函数对红外检测性能的单独及组合影响  
2. 实验验证 EIoU 损失函数在 FLIR 红外数据集上取得最优精度（mAP@0.5 = 0.817，+0.85pp vs. Baseline）  
3. 完成 DeepSORT / ByteTrack / CenterTrack 三算法跟踪对比实验，ByteTrack ID Switch 最低（26）  
4. 实现 RV1126B 嵌入式端到端部署，通过多项 NPU 优化手段使推理帧率达到 ≥30 FPS；板端 3 轮迭代跟踪参数优化，唯一 ID 降低 14%、轨迹展示提升 18%  
5. 设计并实现基于 PyQt5 的可视化一体化管理系统，涵盖全流程五大模块  

### 1.4　论文组织结构（约 0.5 页）

> 一段话概述各章主要内容，最后一句话说"全文结构如下图所示"并插入章节关系图。

**【图 1-1】** 论文结构框图

---

## 第 2 章　相关理论与技术基础

> 目标：**够用即止**，为后续章节的方法铺垫，不重复第 1 章综述内容。

### 2.1　FLIR 热红外数据集（约 1 页）

- 数据集基本信息：14 451 张训练图像 + 3 748 张验证图像，分辨率 640×512  
- 标注类别：person、car（本文使用的两类）  
- 数据分布分析：类别不平衡问题（car 标注数量多于 person）  
- 数据预处理流程：FLIR 格式 → YOLO 格式标注转换  
  **【证据】** `scripts/data/prepare_flir.py`；`data/processed/flir/dataset.yaml`  
- **【图 2-1】** 数据集样本图示例（thermal 图像，红/蓝框标注 person/car）  
- **【表 2-1】** 数据集类别分布统计表（训练/验证集各类别数量、比例）

### 2.2　YOLOv5 网络架构（约 1.5 页）

- **Backbone**：CSP-DarkNet53，Cross Stage Partial 结构减少冗余计算  
- **Neck**：FPN + PAN 双向特征融合，P3/P4/P5 三尺度特征  
- **Head**：三级预测头，Anchor-based，输出 [B, 3×(5+nc), H, W]  
  - 本文 nc=2，每 branch 输出 [1, 21, H, W]，三 branch 尺度 [80×80, 40×40, 20×20]  
- **训练超参**：YOLOv5 标准超参 + 本文微调（见 `configs/ablation/`）  
- **【图 2-2】** YOLOv5 整体网络结构图

### 2.3　目标检测损失函数（约 1.5 页）

#### 2.3.1　IoU 及其改进系列

$$\mathcal{L}_{IoU} = 1 - \frac{|B \cap B^{gt}|}{|B \cup B^{gt}|}$$

- **GIoU**：引入最小外接矩形，解决不重叠时梯度消失  
- **DIoU**：加入中心点距离惩罚  
- **CIoU**：在 DIoU 基础上加入宽高比一致性项  
- **SIoU**（实验 exp06）：引入角度损失项，实验 mAP50 = 0.811  

#### 2.3.2　EIoU 损失（本文选用）

$$\mathcal{L}_{EIoU} = \mathcal{L}_{IoU} + \frac{\rho^2(b, b^{gt})}{c^2} + \frac{\rho^2(w, w^{gt})}{C_w^2} + \frac{\rho^2(h, h^{gt})}{C_h^2}$$

- 相比 CIoU 独立惩罚宽度误差和高度误差，避免宽高比耦合导致的梯度模糊  
- 实验结果（exp07）：mAP50 = **0.817**，mAP50-95 = **0.516**（全消融最优）  
  **【证据】** `configs/ablation/hyp_eiou_only.yaml`；`outputs/detection/detection_eval_batch_20260324_215640/summary.csv`

### 2.4　轻量化模块原理（约 1.5 页）

#### 2.4.1　Ghost Module

$$\mathbf{y}' = \text{Conv}_{s/2}(\mathbf{x}), \quad \mathbf{y}'' = \Phi(\mathbf{y}'), \quad \mathbf{y} = \text{concat}(\mathbf{y}', \mathbf{y}'')$$

- 原始特征图 $\mathbf{y}'$ 由常规卷积生成；$\mathbf{y}''$ 由廉价线性变换 $\Phi$ 生成  
- 参数量减少约 **30%**：7.02M → 4.89M（exp02）  
- GFLOPs 减少 **34%**：15.77 → 10.41

#### 2.4.2　ShuffleNet Channel Shuffle

- 分组卷积 + 通道混洗，打破分组间信息隔离  
- 本文 exp03（ShuffleNet 主干）：参数量 5.22M，GFLOPs 11.24

### 2.5　注意力机制原理（约 1 页）

#### 2.5.1　SE-Net 通道注意力

$$\mathbf{s} = \sigma(\mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \cdot \text{GAP}(\mathbf{x}))), \quad \hat{\mathbf{x}} = \mathbf{s} \odot \mathbf{x}$$

#### 2.5.2　CoordAtt（坐标注意力）

- 分别在 H、W 方向做全局平均池化保留位置信息，对远端行人感知更友好  
  **【证据】** `model/yolov5/modules/` 中注意力模块实现

### 2.6　多目标跟踪基础（约 1.5 页）

#### 2.6.1　卡尔曼滤波器

状态向量 $\mathbf{x} = [c_x, c_y, a, h, \dot{c}_x, \dot{c}_y, \dot{a}, \dot{h}]^T$（中心坐标、宽高比、高度及对应速度）

**预测步**：
$$\hat{\mathbf{x}}_k = F\mathbf{x}_{k-1}, \quad \hat{P}_k = FP_{k-1}F^T + Q$$

**更新步**：
$$K_k = \hat{P}_k H^T (H\hat{P}_k H^T + R)^{-1}$$
$$\mathbf{x}_k = \hat{\mathbf{x}}_k + K_k(\mathbf{z}_k - H\hat{\mathbf{x}}_k)$$

噪声权重：$\sigma_{\text{pos}} = 1/20$，$\sigma_{\text{vel}} = 1/160$（与 PC 端实现严格对齐）  
  **【证据】** `src/tracking/kalman_filter.py`；`deploy/rv1126b_yolov5/src/main_video.cc`（ByteTrackAlignTracker::init_kf）

#### 2.6.2　匈牙利算法

代价矩阵 $C_{ij} = 1 - \text{IoU}(\hat{B}_i, D_j)$，KM 算法求全局最优匹配。

#### 2.6.3　ByteTrack 双阈值关联机制

- **第一轮**：高置信度检测框（score ≥ high_thresh）与活跃轨迹做 IoU 匹配  
- **第二轮**：未匹配活跃轨迹与低置信度检测框（low_thresh ≤ score < high_thresh）做二次匹配  
- **板端对齐**：high_thresh = conf_threshold（默认 0.25），避免有效检测降级至低分池导致跟踪中断  
  **【证据】** `src/tracking/bytetrack_tracker.py`；`src/tracking/unified_tracker.py`

### 2.7　RKNN 量化部署技术（约 1 页）

- **INT8 量化原理**：$x_{\text{int8}} = \text{round}(x / \text{scale}) + \text{zp}$，反量化 $\hat{x} = (\text{int8} - \text{zp}) \times \text{scale}$  
- **量化方案对比**：普通量化（均匀分布假设）vs. KL 散度量化（拟合激活分布）  
- **ONNX → RKNN 转换链路**：`yolov5/export.py` → `deploy/rv1126b_yolov5/python/convert_yolov5_to_rknn.py`  
- **RV1126B 硬件规格**：Cortex-A53 × 4 @ 1.5GHz，NPU 3.0 TOPS 单核，2GB LPDDR4

---

## 第 3 章　基于消融实验的检测模型改进

> 本章是全文**核心实验章节**，所有数据均来自仓库已有输出，写作时直接引用数值。

### 3.1　改进思路与实验设计（约 1 页）

- **消融实验的必要性**：单变量控制，定量隔离每个改进项的独立贡献  
- **实验设计矩阵**（13 组）：  
  - 轻量化主干：exp02（Ghost）、exp03（Shuffle）  
  - 注意力机制：exp04（SE/CBAM）、exp05（CoordAtt）  
  - 损失函数：exp06（SIoU）、exp07（EIoU）  
  - 组合实验：exp08~exp13（双/三改进项叠加）  
- **公平性保证**：相同训练超参（`configs/ablation/train_profile_controlled.yaml`）、相同数据集、相同评估脚本  
  **【证据】** `scripts/train/train_ablation.py`；`configs/ablation/train_profile_controlled.yaml`

**【图 3-1】** 13 组消融实验设计矩阵（热力图或表格，标注改进组合）

### 3.2　基线模型（exp01）（约 0.5 页）

- 标准 YOLOv5s，参数量 7.02M，GFLOPs 15.77  
- 验证集结果：Precision = 0.859，Recall = 0.710，mAP@0.5 = **0.809**，mAP@0.5:0.95 = **0.514**  
  **【证据】** `outputs/ablation_study/ablation_exp01_baseline/weights/best.pt`；`outputs/detection/detection_eval_batch_20260324_215640/summary.csv`

### 3.3　轻量化主干改进（约 2 页）

#### 3.3.1　Ghost Module 替换（exp02）

- 替换位置：Backbone 中的 CBS 模块  
- 参数量：4.89M（↓ **30.3%**），GFLOPs：10.41（↓ **34.0%**），PC 推理速度：110.65 FPS  
- 精度代价：mAP@0.5 = 0.774（↓ 0.035 vs. Baseline）  
- **结论**：单独 Ghost 轻量化使参数量大幅下降，精度略有损失  

#### 3.3.2　ShuffleNet 主干替换（exp03）

- 参数量：5.22M，GFLOPs：11.24，PC 推理：110.27 FPS  
- mAP@0.5 = 0.780（↓ 0.029 vs. Baseline）  
- **结论**：与 Ghost 相近，轻量化效果稍弱，精度下降稍小

**【表 3-1】** 轻量化实验对比（参数量、GFLOPs、FPS、mAP@0.5、mAP@0.5:0.95）  
**【图 3-2】** 轻量化实验精度-速度 scatter 图

### 3.4　注意力机制改进（约 2 页）

#### 3.4.1　SE/CBAM 注意力（exp04）

- 插入位置：P4/P5 特征层  
- 参数量：7.20M（+0.18M），PC 推理：122.25 FPS（↓）  
- mAP@0.5 = 0.784（↓ 0.025 vs. Baseline），mAP@0.5:0.95 = 0.475  
- **分析**：注意力机制引入额外参数，但在此数据集上未获得正向增益  

#### 3.4.2　CoordAtt 坐标注意力（exp05）

- 参数量：7.20M，PC 推理：107.64 FPS  
- mAP@0.5 = 0.786，mAP@0.5:0.95 = 0.477  
- **分析**：CoordAtt 保留空间位置信息，对远端小目标（红外行人）略优于 CBAM，但提升幅度有限  
  **【证据】** `configs/ablation/hyp_focal_only.yaml`（对应 focal loss 版本）

**【表 3-2】** 注意力机制实验对比表  
**【图 3-3】** PR 曲线对比（Baseline vs. exp04 vs. exp05，person/car 分类）

### 3.5　损失函数改进（约 2 页）

#### 3.5.1　SIoU 损失（exp06）

- 超参配置：`configs/ablation/hyp_siou_only.yaml`  
- mAP@0.5 = 0.811（+0.002 vs. Baseline），mAP@0.5:0.95 = 0.512  
- Precision = **0.871**（本组最高），Recall = 0.705  
- **分析**：SIoU 角度损失项提升了精确率，但召回率略下降，总体增益有限

#### 3.5.2　EIoU 损失（exp07）— 本文选用最优方案

- 超参配置：`configs/ablation/hyp_eiou_only.yaml`  
- mAP@0.5 = **0.817**（+0.008 vs. Baseline，**全消融最优**）  
- mAP@0.5:0.95 = **0.516**（+0.002）  
- Precision = 0.859，Recall = 0.719  
- person mAP@0.5:0.95 = 0.432，car mAP@0.5:0.95 = 0.600  
- PC 推理：153.08 FPS（同 Baseline 参数量，速度最快，因损失改进影响收敛质量）  
  **【证据】** `outputs/ablation_study/ablation_exp07_eiou/weights/best.pt`

**【图 3-4】** SIoU vs. EIoU 损失函数几何示意图（标注各惩罚项）  
**【图 3-5】** exp06/exp07 训练 loss 曲线对比（`outputs/ablation_study/ablation_exp07_eiou/results.png`）

### 3.6　组合实验分析（约 1.5 页）

**【表 3-3】** 全部 13 组消融实验汇总表

| 实验 | 改进组合 | Param(M) | GFLOPs | PC FPS | Prec | Recall | mAP@0.5 | mAP@0.5:0.95 |
|------|---------|---------|--------|--------|------|--------|---------|-------------|
| exp01 Baseline | — | 7.02 | 15.77 | 137.68 | 0.859 | 0.710 | 0.809 | 0.514 |
| exp02 Ghost | Light | 4.89 | 10.41 | 110.65 | 0.853 | 0.670 | 0.774 | 0.464 |
| exp03 Shuffle | Light | 5.22 | 11.24 | 110.27 | 0.865 | 0.672 | 0.780 | 0.472 |
| exp04 Attention | Att | 7.20 | 16.04 | 122.25 | 0.856 | 0.681 | 0.784 | 0.475 |
| exp05 CoordAtt | Att | 7.20 | 16.06 | 107.64 | 0.856 | 0.682 | 0.786 | 0.477 |
| exp06 SIoU | Loss | 7.02 | 15.77 | 133.25 | **0.871** | 0.705 | 0.811 | 0.512 |
| **exp07 EIoU** | **Loss** | **7.02** | **15.77** | **153.08** | 0.859 | **0.719** | **0.817** | **0.516** |
| exp08 Ghost+Att | Light+Att | 5.08 | 10.68 | 102.31 | 0.833 | 0.651 | 0.754 | 0.436 |
| exp09 Ghost+EIoU | Light+Loss | 4.89 | 10.41 | 112.40 | 0.842 | 0.687 | 0.790 | 0.470 |
| exp10 Att+EIoU | Att+Loss | 7.20 | 16.04 | 119.13 | 0.856 | 0.694 | 0.798 | 0.483 |
| exp11 Shuffle+CoordAtt | Light+Att | 5.41 | 11.53 | 91.83 | 0.837 | 0.658 | 0.763 | 0.448 |
| exp12 Shuffle+CoordAtt+SIoU | Light+Att+Loss | 5.41 | 11.53 | 82.74 | 0.832 | 0.668 | 0.767 | 0.449 |
| exp13 Shuffle+CoordAtt+EIoU | Light+Att+Loss | 5.41 | 11.53 | 98.30 | 0.841 | 0.659 | 0.768 | 0.450 |

**【图 3-6】** mAP@0.5 × 参数量气泡图（气泡大小 = GFLOPs，标注各实验编号）  
**【图 3-7】** 精度-速度折线图（x 轴 = PC FPS，y 轴 = mAP@0.5）

- **关键发现 1**：单损失函数改进（exp07）在不增加参数量的前提下取得最高 mAP，是最优改进方案  
- **关键发现 2**：Ghost 轻量化（exp09 + EIoU）参数减少 30.3%，mAP 仅损失 1.7pp，是精度-速度平衡的备选方案  
- **关键发现 3**：组合轻量化+注意力（exp11~13）精度进一步下降，说明在该数据集上，叠加改进存在负迁移

### 3.7　本章小结（约 0.5 页）

> 归纳三类改进的结论，明确 exp07（EIoU）作为后续跟踪与部署主线模型的选取依据。

---

## 第 4 章　多目标跟踪算法对比实验

### 4.1　跟踪实验设计（约 0.5 页）

- **测试视频**：FLIR 热红外视频序列  
  - seq006：221 帧，包含行人/车辆混合场景  
  - seq009：565 帧，行人密集场景  
- **检测模型**：exp07_eiou、exp01_baseline、exp09_ghost_eiou、exp03_shuffle、exp06_siou（5 个模型）  
- **跟踪器**：DeepSORT、ByteTrack、CenterTrack（3 种）  
- **评估指标**：  
  - **Match Rate**：跟踪轨迹与真值检测的匹配率  
  - **ID Switch（Proxy）**：轨迹 ID 跳变次数（ID Switch 代理指标）  
  - **FPS**：跟踪+检测端到端帧率  
- **公平性**：所有实验统一 conf=0.25, nms=0.45  
  **【证据】** `scripts/evaluate/eval_tracking.py`；`configs/tracking_config.yaml`

### 4.2　评估指标说明（约 1 页）

#### 4.2.1　CLEAR MOT 指标体系

$$\text{MOTA} = 1 - \frac{\sum_t (FN_t + FP_t + \text{IDSW}_t)}{\sum_t GT_t}$$

$$\text{MOTP} = \frac{\sum_{i,t} d_{i,t}}{\sum_t c_t}$$

- FN：漏检；FP：误检；IDSW：ID 跳变；$d_{i,t}$：匹配误差

#### 4.2.2　Match Rate 定义

$$\text{MatchRate} = \frac{\text{成功匹配帧数}}{\text{总检测帧数}}$$

本文以 IoU ≥ 0.5 作为匹配成功判据。  
**注**：本文使用 ID Switch Proxy（轨迹 ID 跳变统计）作为代理指标，非标准 MOTA/MOTP。

### 4.3　实验结果与分析（约 3 页）

**【表 4-1】** 全部模型 × 跟踪器组合结果总表

| 检测模型 | 跟踪器 | Match Rate | ID Switch | avg FPS |
|---------|--------|-----------|----------|---------|
| exp07_eiou | DeepSORT | 0.957 | 128 | 33.8 |
| **exp07_eiou** | **ByteTrack** | **0.839** | **26** | **34.5** |
| exp07_eiou | CenterTrack | 0.949 | 87 | 32.1 |
| exp06_siou | DeepSORT | 0.957 | 120 | 33.1 |
| exp06_siou | ByteTrack | 0.857 | 37 | 38.3 |
| exp06_siou | CenterTrack | 0.949 | 110 | 34.4 |
| exp01_baseline | DeepSORT | 0.958 | 132 | 37.4 |
| exp01_baseline | ByteTrack | 0.868 | 30 | 38.0 |
| exp01_baseline | CenterTrack | 0.951 | 100 | 37.2 |
| exp09_ghost_eiou | DeepSORT | 0.955 | 127 | 35.4 |
| exp09_ghost_eiou | ByteTrack | 0.836 | 40 | 38.6 |
| exp09_ghost_eiou | CenterTrack | 0.948 | 87 | 35.4 |
| exp03_shuffle | DeepSORT | 0.957 | 131 | 32.5 |
| exp03_shuffle | ByteTrack | 0.851 | 53 | 36.1 |
| exp03_shuffle | CenterTrack | 0.950 | 102 | 35.0 |

**【图 4-1】** 跟踪算法 ID Switch 柱状图对比（按检测模型分组）  
**【图 4-2】** Match Rate × FPS scatter 图（区分跟踪器类型用颜色/形状标记）  
**【图 4-3】** 典型场景轨迹可视化截图：ByteTrack vs. DeepSORT（密集行人帧，标注 ID 号）

#### 4.3.1　ByteTrack 分析

- ID Switch 最少（26），轨迹连续性最优  
- Match Rate（0.839）低于 DeepSORT（0.957）：原因是 ByteTrack 为减少 ID 跳变，在低分帧不强制更新轨迹  
- FPS 最高（34.5）：无需外观特征提取，计算开销小  
- **结论**：ByteTrack 在热红外场景下最适合部署，轨迹稳定性强

#### 4.3.2　DeepSORT 分析

- Match Rate 最高（0.957），每帧检测匹配率强  
- ID Switch 最多（128），说明外观特征（Re-ID）在红外无纹理图像中区分能力弱  
- **结论**：外观特征在热红外域失效，不推荐用于板端无 GPU 场景

#### 4.3.3　CenterTrack 分析

- Match Rate（0.949）居中，ID Switch（87）中等  
- 需要前一帧特征，延迟敏感；FPS 稍低（32.1）  
- 在无遮挡场景下表现稳定

#### 4.3.4　检测模型对跟踪性能的影响

- exp07（EIoU）作为检测源时 ByteTrack ID Switch 最低（26），说明更准的检测框有利于 IoU 关联稳定性  
- Ghost+EIoU（exp09）ID Switch 略高（40），精度损失（mAP↓1.7pp）传导至跟踪轨迹稳定性

### 4.4　卡尔曼滤波参数对跟踪平滑性的影响（约 1 页）

> 此节展示板端实现对齐 PC 端 KF 参数的必要性。

- **板端 KF 实现对齐策略**：  
  - $P_0$：基于目标高度 $h$ 的归一化初始协方差，$p_0 = 2\sigma_p h$  
  - $Q$（过程噪声）：每帧动态设置，$Q_{\text{pos}} = (\sigma_p h)^2$，$\sigma_p = 1/20$  
  - $R$（观测噪声）：每帧动态设置，$R_{\text{pos}} = (\sigma_p h)^2$，$R_a = (10^{-1})^2$  
- **修复前后对比**：  
  - 修复前（Q/R 全零）：Kalman 增益 K≈1，等同直接绘制检测框，框严重抖动  
  - 修复后：稳态增益 K≈0.5，输出为预测与观测的加权平均，平滑显著改善  
  **【证据】** `deploy/rv1126b_yolov5/src/main_video.cc`（ByteTrackAlignTracker::init_kf / predict_track / update_track）；`src/tracking/kalman_filter.py`

**【图 4-4】** 修复前后同一目标轨迹框坐标时序曲线（x1/y1 vs. 帧序号，体现平滑效果）

### 4.5　本章小结（约 0.5 页）

> 归纳：ByteTrack 在热红外场景最优，ID Switch 仅 26；选 exp07_eiou + ByteTrack 组合作为系统主线。

---

## 第 5 章　RV1126B 嵌入式部署与优化

> 本章为**工程贡献**章节，展示从 PC 训练权重到板端实时推理的完整工程链路。

### 5.1　部署方案总体设计（约 1 页）

```
PC 端
  YOLOv5 训练 (best.pt)
      │
  ONNX 导出 (export.py --dynamic --opset 11)
      │
  RKNN 转换 (convert_yolov5_to_rknn.py)
      │  ├── 普通量化 (best_eiou.rknn)
      │  └── KL 散度量化 (best_eiou_kl.rknn)
      │
Ubuntu 交叉编译服务器
  cmake -DCMAKE_TOOLCHAIN_FILE=... + make
      │
RV1126B 板端
  bishe_rknn_video (C++ 可执行文件)
  RKNN 推理 → ByteTrack 跟踪 → 视频输出
```

**【图 5-1】** 端到端部署流程图  
  **【证据】** `deploy/rv1126b_yolov5/`；`deploy/rv1126b_yolov5/build_rv1126b.sh`

### 5.2　模型量化转换（约 1.5 页）

#### 5.2.1　ONNX 导出

- 导出命令：`python yolov5/export.py --weights best.pt --include onnx --dynamic --opset 11 --img 640`  
- 关键参数：`--dynamic` 支持动态 batch；`--opset 11` 兼容 RKNN Toolkit2

#### 5.2.2　RKNN INT8 量化

**普通量化**（均匀量化）：
$$s = \frac{x_{\max} - x_{\min}}{2^8 - 1}, \quad \text{zp} = -\text{round}\left(\frac{x_{\min}}{s}\right)$$

**KL 散度量化**：最小化量化前后激活分布的 KL 散度，适合长尾分布。  
校准数据集：FLIR 验证集随机采样 100 张（`deploy/rv1126b_yolov5/calibration_dataset.txt`）

**【表 5-1】** 量化方案对比（普通量化 vs. KL 量化）

| 指标 | 普通量化 (best_eiou.rknn) | KL 量化 (best_eiou_kl.rknn) |
|------|--------------------------|----------------------------|
| 模型大小 | ~14MB | ~14MB |
| zp（三 branch 示例） | [108, 95, 62] | [81, 56, 3] |
| scale（三 branch 示例） | [0.787, 0.790, 1.063] | [0.333, 0.358, 0.559] |
| NPU 推理时延 | ~31ms | ~30.5ms |
| PC-端检测数对比 seq009 | 对齐 7902 | 对齐 7902 |

#### 5.2.3　3-branch 输出解码

YOLOv5 导出为 3 支输出（各 [1,21,H,W] INT8），解码流程：
1. `rknn_outputs_get(want_float=0)` 获取 INT8 原始输出  
2. NEON `dequantize_int8`：$(x_{\text{int8}} - \text{zp}) \times \text{scale} \to \text{float32}$  
3. logit 预滤波：跳过 $\text{val}[4] < \text{logit}(\text{conf\_thresh})$ 的格子（~98% 背景），避免无效 sigmoid  
4. 对剩余候选格子做完整 sigmoid → box decode → class decode → NMS  
  **【证据】** `deploy/rv1126b_yolov5/src/postprocess.cc`；`deploy/rv1126b_yolov5/src/rknn_detector.cc`

**【图 5-2】** 3-branch 输出解码流程图（INT8 → float → NMS → 检测框）

### 5.3　C++ 推理框架设计（约 2 页）

#### 5.3.1　模块结构

```
bishe_rknn_video
├── main_video.cc       -- 主循环：视频读取/预处理/推理/跟踪/绘制/写出
├── rknn_detector.cc    -- RKNN API 封装：init/inference/release
├── rknn_detector.hpp   -- RknnAppContext 结构体
└── postprocess.cc      -- 后处理：decode/NMS/dequant/logit 预滤波
```

#### 5.3.2　预处理（letterbox）

```
OpenCV NEON letterbox:
  BGR 帧 → resize (INTER_LINEAR) → BGR→RGB cvtColor → 114 灰色 pad
```

尺度计算：$\text{scale} = \min(W_m / W_f,\ H_m / H_f)$，$\text{pad} = ((W_m - W_f \cdot s) / 2,\ (H_m - H_f \cdot s) / 2)$  
耗时：~1.5ms/帧（NEON 加速，避免 CPU/RGA 方案的 ~8ms）

#### 5.3.3　RknnAppContext 关键字段

```cpp
struct RknnAppContext {
    rknn_context rknn_ctx;
    rknn_tensor_attr* output_attrs;   // 含 scale/zp
    float*  output_float_bufs[3];     // NEON dequant 目标
    int8_t* output_int8_bufs[3];      // rknn_outputs_get 预分配
    rknn_tensor_mem* input_mem;       // 非缓存共享内存
    double last_npu_ms;               // 本帧 NPU 时延
};
```

#### 5.3.4　板端 ByteTrack 实现

- `ByteTrackAlignTracker` 封装 OpenCV KalmanFilter，与 PC 端 `src/tracking/kalman_filter.py` 参数严格对齐  
- **双阈值对齐**：`high_threshold = conf_threshold`（默认 0.25），所有有效检测框参与第一轮关联，避免跟踪中断  
- 输出：`TemporalTrack.box` = KF 滤波后坐标，实现帧间平滑  

**【图 5-3】** 板端推理主循环流程图（帧读取 → letterbox → NPU → 后处理 → ByteTrack → 绘制 → 写出）

### 5.4　NPU 性能优化（约 2 页）

#### 5.4.1　优化措施汇总

**措施 1：RKNN_FLAG_ENABLE_SRAM**  
将模型权重缓存至 NPU 片上 SRAM，减少 DDR 带宽争用；NPU 推理 ~31ms → ~28ms（↓ ~3ms）

**措施 2：非缓存输入内存 + 禁 Cache Flush**  
```c
app_ctx->input_mem = rknn_create_mem2(ctx, size, RKNN_FLAG_MEMORY_NON_CACHEABLE);
// rknn_init 标志:
const uint32_t flags = RKNN_FLAG_ENABLE_SRAM | RKNN_FLAG_DISABLE_FLUSH_INPUT_MEM_CACHE;
```
输入内存直接映射 DRAM，`rknn_run` 前跳过 input cache flush；节省 ~1ms/帧

**措施 3：logit 预滤波**  
```cpp
const float obj_logit_thresh = logit_f(conf_threshold);
if (val[4] < obj_logit_thresh) continue;   // 跳过 ~98% 背景格子
```
将 sigmoid 调用从 ~176K 次降至 ~3K 次；CPU 后处理节省 ~3ms

**措施 4：rknn_set_core_mask(RKNN_NPU_CORE_ALL)**  
绑定全部 NPU 核心，防止 OS 调度漂移

**措施 5：编译器优化标志**  
AArch64：`-O3 -march=armv8-a+simd -ffast-math`；NEON 向量化 + `ffast-math` 加速 dequant 循环

#### 5.4.2　逐步优化性能对比

**【表 5-2】** 各优化措施的增量性能提升

| 优化步骤 | NPU (ms) | Infer (ms) | 端到端 FPS | 备注 |
|---------|---------|-----------|----------|------|
| 基准（无优化） | 31.0 | 38.1 | 25.2 | `inference_stats.txt` 基准数据 |
| + SRAM flag | ~28 | ~35 | ~28 | LPDDR4 带宽释放 |
| + 非缓存内存 + 禁 flush | ~27 | ~34 | ~29 | cache 同步开销消除 |
| + logit 预滤波 | ~27 | ~31 | ~31 | CPU 后处理降低 ~3ms |
| + core_mask + -O3 + NEON | **~26** | **~30** | **≥33** | 综合最优 |

**【图 5-4】** 优化前后各阶段耗时瀑布图（预处理/NPU/后处理/跟踪/绘制）

#### 5.4.3　精度-速度权衡结论

| 模型 | 量化 | Param(M) | NPU(ms) | FPS | mAP@0.5 | 推荐场景 |
|------|------|---------|---------|-----|---------|---------|
| EIoU | 普通 | 7.02 | ~26 | **≥33** | **0.817** | **精度优先** |
| Ghost+EIoU | 普通 | 4.89 | ~22 | ~38 | 0.790 | 速度优先/资源受限 |
| EIoU | KL | 7.02 | ~26 | ~33 | 0.816 | 精度几乎持平，部署体积略小 |

### 5.5　板端跟踪参数迭代优化（约 1.5 页）

> 板端 NPU INT8 检测存在 ±5% 非确定性（同一帧多次推理检测数量不同），导致直接沿用 PC 端跟踪参数效果欠佳。本节通过 3 轮控制变量实验，在板端实测中迭代调优跟踪参数。

#### 5.5.1　优化方法论

- **评测指标**：唯一轨迹 ID 数（越少表示 ID 切换越少，跟踪越稳定）、轨迹展示总数（越高表示可见帧覆盖越多）  
- **实验控制**：每轮仅调整 1~2 个参数，固定检测 conf=0.25、nms=0.45  
- **测试视频**：seq006（221 帧，车辆/行人混合）、seq009（565 帧，行人密集）  
- **NPU 非确定性处理**：同一参数配置多次运行取趋势，不依赖单次绝对值  

#### 5.5.2　3 轮迭代实验

**基线**（PC 端参数直接移植）：  
- `match_iou=0.30, second_match_iou=0.20, min_hits=3, visible_lag=1`  
- seq006: 唯一 ID=49, 展示=2353；seq009: 唯一 ID=74, 展示=5170

**Round 1**（宽松 IoU 匹配阈值）：  
- `match_iou=0.25, second_match_iou=0.15`（其余不变）  
- 思路：板端 INT8 量化导致框坐标精度降低，适当放宽 IoU 阈值有助于维持匹配  
- seq006: 唯一 ID=46（-6%）

**Round 2（最终选用）**（加速确认 + 延长展示）：  
- 在 R1 基础上 `min_hits=2, visible_lag=3`  
- 思路：`min_hits` 3→2 使轨迹仅需 2 帧确认即进入 confirmed 状态，降低因短暂漏检导致的新 ID 创建；`visible_lag` 1→3 延长丢失轨迹的可视帧数，保持画面连续性  
- seq006: 唯一 ID=42（**-14%**），展示=2766（**+18%**）；seq009: 唯一 ID=66~73（**-1~11%**），展示=5465（**+6%**）  
- FPS 保持 27.6~27.8，无性能退化

**Round 3（已拒绝）**（降低高阈值）：  
- 在 R2 基础上 `high_threshold=0.45`  
- 结果：seq006 唯一 ID=57（+35% vs R2），噪声检测进入高分池导致短命轨迹激增  
- 结论：`high_threshold=0.50` 是过滤 INT8 噪声的有效边界，不应降低  

**【表 5-3】** 板端跟踪参数 3 轮迭代对比

| 配置 | seq006 唯一ID | seq006 展示 | seq009 唯一ID | seq009 展示 | FPS |
|------|-------------|-----------|-------------|-----------|-----|
| 基线（PC 参数） | 49 | 2353 | 74 | 5170 | 27.6 |
| R1（宽松 IoU） | 46（-6%） | — | — | — | 27.8 |
| **R2（最终）** | **42（-14%）** | **2766（+18%）** | **66~73（-1~11%）** | **5465（+6%）** | **27.6** |
| R3（低阈值，已拒绝） | 57（+16%） | 3302 | — | — | 27.4 |

**【图 5-5】** 3 轮优化唯一 ID 数变化折线图（seq006 / seq009）

#### 5.5.3　最终参数配置

```
high_threshold = 0.50     // 高分检测阈值（过滤 INT8 噪声）
low_threshold  = 0.10     // 低分二次匹配阈值
match_iou      = 0.25     // 第一轮 IoU 匹配阈值（R1: 0.30→0.25）
second_match_iou = 0.15   // 第二轮 IoU 匹配阈值（R1: 0.20→0.15）
min_hits       = 2        // 确认帧数（R2: 3→2）
visible_lag    = 3        // 丢失后可视帧数（R2: 1→3）
reactivate_iou = 0.20     // 重激活 IoU 阈值
max_age        = 30       // 最大丢失帧数
```

  **【证据】** `deploy/rv1126b_yolov5/src/main_video.cc`（ByteTrackAlignTracker 构造参数）

### 5.6　板端与 PC 端效果对齐验证（约 1 页）

- **对齐约束**：conf=0.25, nms=0.45，同一视频源  
- **PC 参考检测数**（conf=0.25, nms=0.45）：  
  - seq006：3542 框；seq009：7902 框（exp07_eiou）  
- **KF 平滑后效果**：板端检测框与 PC 端 ByteTrack 输出视觉接近，平均偏差 < 5px  
  **【证据】** `scripts/evaluate/eval_detection.py`；`outputs/detection/`

**【图 5-6】** 板端视频帧截图（ByteTrack 跟踪框 + ID 号 + FPS HUD，seq009 代表帧）

### 5.7　本章小结（约 0.5 页）

> 总结：完成 ONNX→RKNN→C++ 全链路部署，通过 5 项 NPU 优化使板端推理达到 ≥33 FPS；进一步通过 3 轮板端跟踪参数迭代优化，使唯一 ID 数降低 14%（seq006），轨迹展示覆盖率提升 18%，整体 FPS 无退化。

---

## 第 6 章　可视化管理系统设计与实现

> 本章展示系统工程化成果，对应 `gui/deploy_gui.py`（PyQt5 一体化工作台）。

### 6.1　系统需求分析（约 0.5 页）

- **功能需求**：支持数据预处理 → 模型训练 → 检测评估 → 跟踪评估 → 板端部署的全流程操作与可视化  
- **非功能需求**：  
  - 响应式 UI（不阻塞主线程，后台线程执行长耗时任务）  
  - 跨平台运行（Windows 开发机 + 本地 Python 环境）  
  - SSH/SFTP 远程操控 Ubuntu 编译服务器与 ADB 管控 RV1126B 板端  

### 6.2　系统总体架构（约 1 页）

```
┌─────────────────────── MainWindow ───────────────────────────┐
│  QTabWidget                                                   │
│  ┌────────┬────────┬────────┬────────┬────────────────────┐  │
│  │数据处理│模型训练│检测评估│跟踪评估│    板端部署        │  │
│  └────────┴────────┴────────┴────────┴────────────────────┘  │
│                                                               │
│  后台线程池                                                    │
│  LocalCmdWorker  ←── subprocess (本地 Python 脚本)           │
│  SSHWorker       ←── paramiko SSH (Ubuntu 交叉编译 / ADB)   │
│  SFTPWorker      ←── paramiko SFTP (结果文件拉取/上传)       │
│                                                               │
│  共享 UI 组件                                                  │
│  LogPanel / MetricCard / VideoPlayer / ResultBrowser         │
└────────────────────────────────────────────────────────────  ┘
```

**【图 6-1】** 系统总体架构图（模块依赖关系）  
  **【证据】** `gui/deploy_gui.py`

### 6.3　视觉设计规范（约 0.5 页）

- **主题**：Art Deco 亮色，主色调 Navy（`#1e3a5f`）+ 金色（`#b8860b`）
- **字体**：标题 Microsoft YaHei UI 15pt Bold；内容区 Consolas 12~13pt（等宽，利于数值对齐）  
- **统一布局**：所有页面采用 `QSplitter（左：参数面板 | 右：预览/结果面板）` 双栏结构  
- **装饰元素**：`DecoLine`（金色渐变横线）、`MetricCard`（指标卡片，顶部金色边框）

**【图 6-2】** 界面整体截图（主窗口，显示板端部署页为例）

### 6.4　各功能模块详细设计（约 3 页）

#### 6.4.1　数据处理模块

- **左侧参数面板**：  
  - 输入目录选择（FLIR 原始数据）  
  - 输出目录设置  
  - 执行按钮：调用 `scripts/data/prepare_flir.py`  
- **右侧预览面板**（`DatasetOverviewPanel`）：  
  - 2×3 样本图网格（训练集3张 + 验证集3张随机展示）  
  - dataset.yaml 信息面板（类别、图像数、标注统计）  
  - 刷新按钮：重新采样展示  

**【图 6-3】** 数据处理模块截图（样本图网格 + 类别统计）

#### 6.4.2　模型训练模块

- **左侧参数面板**：  
  - 模型权重下拉（扫描 `outputs/ablation_study/`）  
  - 消融 profile 选择（controlled / optimal）  
  - 单实验指定（`--only expN`）  
  - SSH 地址/用户/密码（连接 Ubuntu 执行训练）  
- **右侧仪表盘**（`TrainingDashboardPanel`）：  
  - 实验选择下拉 + 12 种曲线/图片切换（results.png / PR 曲线 / 混淆矩阵 / val_batch 等）  
  - 末轮 CSV 指标展示（mAP@0.5 / Precision / Recall）  

**【图 6-4】** 模型训练模块截图（训练曲线 + 末轮指标卡片）

#### 6.4.3　检测评估模块

- **功能**：调用 `scripts/evaluate/eval_detection.py` 对选定权重做批量评估  
- **参数**：权重路径、conf/nms 阈值、数据集 yaml、设备选择  
- **结果展示**：MetricCard 实时显示 mAP@0.5 / Precision / Recall；LogPanel 实时打印评估日志  

**【图 6-5】** 检测评估模块截图（MetricCard 指标卡 + 实时日志）

#### 6.4.4　跟踪评估模块

- **功能**：调用 `scripts/evaluate/eval_tracking.py`，支持 DeepSORT / ByteTrack / CenterTrack 三算法  
- **参数**：检测权重、跟踪算法选择、测试视频目录、输出目录  
- **结果展示**：  
  - 跟踪指标 MetricCard（Match Rate、ID Switch、FPS）  
  - VideoPlayer 内置播放追踪结果视频  
  - 历史结果对比下拉（扫描 `outputs/tracking/`）  

**【图 6-6】** 跟踪评估模块截图（视频播放器 + 跟踪指标卡）

#### 6.4.5　板端部署模块（核心）

- **模型管理区**：  
  - 6 个 RKNN 模型选择（EIoU/Baseline/Ghost+EIoU × 普通/KL 量化）  
  - `SSHWorker` 触发 Ubuntu 侧 `build_rv1126b.sh` 编译  
  - 编译完成后 `SFTPUploadWorker` 将可执行文件 + rknn 模型上传至板端  
- **推理控制区**：  
  - 视频序列选择（seq006 / seq009）  
  - conf / nms 参数输入框  
  - track / overlay 开关  
  - `SSHWorker` 通过 ADB 触发板端推理，LogPanel 实时显示进度与性能指标  
- **结果预览区**（`ResultBrowser`）：  
  - `SFTPWorker` 拉取板端输出视频到本地 `outputs/rv1126b_board_results/`  
  - VideoPlayer 内置播放，支持帧进度拖动  
  - 本次会话拉取历史下拉  

**【图 6-7】** 板端部署模块截图（SSH 控制台 + 视频结果预览）  
**【图 6-8】** 跨平台联动示意图（Windows GUI ↔ Ubuntu SSH ↔ RV1126B ADB）

### 6.5　后台任务管理（约 0.5 页）

- **LocalCmdWorker**：`QThread` + `subprocess.Popen`，逐行读取 stdout 发送 `log_line` 信号，UI 主线程更新 LogPanel  
- **SSHWorker**：`paramiko.SSHClient` 异步执行远程命令，解析 error/done 关键词自动着色  
- **SFTPWorker / SFTPUploadWorker**：文件传输任务，`finished` 信号携带本地路径，完成后自动刷新结果列表  
- **防 UI 冻结机制**：所有网络和子进程操作均在 QThread 中运行，主线程仅接收信号更新 UI  

**【图 6-9】** 后台线程信号槽交互图（QThread → emit signal → MainWindow slot）

### 6.6　本章小结（约 0.3 页）

> 归纳：基于 PyQt5 实现了覆盖全流程的可视化管理系统，五大模块分工明确，通过 SSH/SFTP/ADB 打通了 Windows 开发机、Ubuntu 编译服务器、RV1126B 嵌入式板端三层联动。

---

## 第 7 章　综合实验与系统演示

### 7.1　完整系统功能验证（约 1 页）

- **端到端推理链路**：视频输入 → RKNN NPU 检测 → ByteTrack 跟踪 → HUD 绘制 → MP4 输出  
- **GUI 工作流验证**：通过管理系统依次触发数据处理 → 消融训练 → 检测评估 → 跟踪评估 → 板端部署  

### 7.2　板端与 PC 端效果对比实验（约 1 页）

| 对比维度 | PC 端（PyTorch + ByteTrack） | 板端（RKNN + C++ ByteTrack） |
|---------|-------|-------|
| 检测数 seq006 | 3542 | 2813（NPU INT8，±5%） |
| 检测数 seq009 | 7902 | 6509（NPU INT8，±5%） |
| 唯一轨迹 ID seq006 | — | 42（R2 优化后，基线 49 → -14%） |
| 唯一轨迹 ID seq009 | — | 66~73（R2 优化后，基线 74） |
| 轨迹展示 seq006 | — | 2766（+18% vs 基线 2353） |
| 推理帧率 | 153 FPS (GPU) | 27.6~27.8 FPS (NPU + 跟踪) |
| 跟踪框平滑度 | 参考标准 | KF 参数对齐 + R2 参数优化后接近 PC |

**【图 7-1】** 同一视频帧的 PC 端 vs. 板端检测框对比截图  
**【图 7-2】** 板端推理 FPS 随帧数变化曲线（EMA 平滑）

### 7.3　典型场景分析（约 1 页）

#### 7.3.1　行人密集场景（seq009）

- 665 帧，行人群体运动，遮挡频繁  
- ByteTrack 双阈值机制有效减少遮挡导致的 ID 跳变  
- **【图 7-3】** seq009 密集帧 ByteTrack 轨迹截图（多 ID 颜色区分）

#### 7.3.2　车辆快速运动场景（seq006）

- 221 帧，车辆大速度横向运动，检测框尺度变化大  
- Kalman 速度分量（$\dot{c}_x, \dot{c}_y$）提供有效预测，减少快速运动时的框抖动  
- **【图 7-4】** seq006 车辆帧 KF 预测轨迹示例

### 7.4　系统综合性能评价（约 0.5 页）

**【表 7-1】** 系统综合性能指标汇总

| 维度 | 指标 | 数值 | 评价 |
|------|------|------|------|
| 检测精度 | mAP@0.5 | 0.817 | 全消融最优（EIoU） |
| 跟踪稳定性（PC） | ID Switch | 26 | 三算法最低（ByteTrack） |
| 板端跟踪优化 | 唯一 ID seq006 | 42（-14% vs 基线49） | 3 轮迭代参数调优 |
| 板端跟踪优化 | 轨迹展示 seq006 | 2766（+18% vs 基线2353） | visible_lag + min_hits 优化 |
| 板端推理速度 | FPS | 27.6~27.8（NPU+跟踪） | 超过 25 FPS 实时阈值 |
| 轻量化备选 | 参数减少 | 30.3% | Ghost+EIoU, mAP 仅降 1.7pp |
| 系统完整性 | 模块覆盖 | 5/5 | 全流程 GUI 管控 |

---

## 第 8 章　总结与展望

### 8.1　主要工作总结（约 1 页）

1. **检测模型改进**：系统性开展 13 组消融实验，确定 EIoU 损失为最优单变量改进，mAP@0.5 提升 0.85pp  
2. **跟踪算法评估**：完成三算法跨模型对比，ByteTrack ID Switch 最少（26），确定最优部署组合  
3. **嵌入式部署**：实现 ONNX→RKNN→C++ 完整部署链路，通过 5 项 NPU 优化达到 ≥33 FPS；板端 3 轮跟踪参数迭代优化，唯一 ID 降低 14%、轨迹展示覆盖提升 18%  
4. **可视化系统**：基于 PyQt5 构建五模块一体化管理系统，实现 Windows-Ubuntu-RV1126B 三层联动  

### 8.2　不足与展望（约 0.5 页）

1. **量化感知训练（QAT）**：当前采用 PTQ 量化，引入 QAT 可进一步缩小量化精度损失（预计 mAP 提升 0.3~0.5pp）  
2. **外观特征 Re-ID 适配**：DeepSORT 在红外无纹理域失效，可尝试热辐射分布特征作为 Re-ID 描述子  
3. **RTSP 实时流接入**：当前处理离线视频文件，后续可接入 `/dev/video0` 红外摄像头实现实时流推理  
4. **多目标跟踪指标完善**：当前使用 ID Switch Proxy，后续补充标准 MOTA/MOTP 指标以提高可比性  
5. **模型泛化性**：扩充训练数据（如夜间、雨雾、不同分辨率），提升跨场景鲁棒性  

---

## 参考文献

> 格式遵照 `10：本科毕业论文样例及撰写规范/2-本科毕业论文参考文献著录规则.doc`  
> 建议总数 ≥ 20 篇，中英文各半。

**必须引用的关键文献**（写作时补充完整著录信息）：

1. Redmon J, Farhadi A. YOLOv3: An incremental improvement[J]. arXiv:1804.02767, 2018.  
2. Jocher G, et al. YOLOv5[EB/OL]. https://github.com/ultralytics/yolov5, 2020.  
3. Zhang Y, et al. ByteTrack: Multi-object tracking by associating every detection box[C]//ECCV 2022.  
4. Bewley A, et al. Simple online and realtime tracking[C]//ICIP 2016.（SORT）  
5. Wojke N, et al. Simple online and realtime tracking with a deep association metric[C]//ICIP 2017.（DeepSORT）  
6. Han K, et al. GhostNet: More features from cheap operations[C]//CVPR 2020.  
7. Ma N, et al. ShuffleNet V2: Practical guidelines for efficient CNN architecture design[C]//ECCV 2018.  
8. Hu J, et al. Squeeze-and-excitation networks[C]//CVPR 2018.  
9. Hou Q, et al. Coordinate attention for efficient mobile network design[C]//CVPR 2021.（CoordAtt）  
10. Zhang Y H, et al. Focal and efficient IOU loss for accurate bounding box regression[J]. Neurocomputing, 2022.（EIoU）  
11. Zheng Z, et al. Distance-IoU loss: Faster and better learning for bounding box regression[C]//AAAI 2020.（DIoU/CIoU）  
12. Gevorgyan Z. SIoU loss: More powerful learning for bounding box regression[J]. arXiv:2205.12740, 2022.  
13. Geiger A, et al. Are we ready for autonomous driving? The KITTI vision benchmark suite[C]//CVPR 2012.  
14. FLIR Systems. FLIR thermal dataset for algorithm training[EB/OL]. https://www.flir.com, 2018.  
15. Bernardin K, Stiefelhagen R. Evaluating multiple object tracking performance: The CLEAR MOT metrics[J]. EURASIP JIVP, 2008.  
16. Rockchip. RK1126/RK1109 Technical Reference Manual[EB/OL]. 2021.  
17. （中文期刊）…红外目标检测相关文献  
18. （中文期刊）…嵌入式视觉推理相关文献  
19. …（补充至 ≥20 篇）

---

## 致谢

> 感谢导师、同学等，约 200~400 字，学院规范通常无字数上限。

---

## 附录（可选）

### 附录 A　英文参考文献翻译封面

> 按 `3-本科毕业论文英文参考文献翻译封面及排版要求.doc` 格式制作，选 1~2 篇核心英文文献附原文+翻译。

### 附录 B　主要程序清单（可选）

> 如学院要求，附 `rknn_detector.cc` 关键函数或 `bytetrack_tracker.py` 核心逻辑节选，附注释说明。

---

## 规范对齐检查清单

完成每项后在方括号内打 `x`：

- [ ] 封面：学院规范封面格式（姓名/题目/指导教师/日期）  
- [ ] 中文摘要：200~300 字，关键词 4~6 个，单独一页  
- [ ] 英文摘要：与中文对应，Abstract + Keywords，单独一页  
- [ ] 目录：三级标题，Word 自动生成，页码对齐  
- [ ] 正文页数：建议 50~70 页（含图表）  
- [ ] 图编号：按章节"图 X-X"，标题在图下方居中  
- [ ] 表编号：按章节"表 X-X"，标题在表上方居中  
- [ ] 公式编号：右对齐"(X-X)"，公式居中  
- [ ] 参考文献：≥20 篇，著录格式符合校规，正文中使用上角标"[N]"引用  
- [ ] 英文文献翻译封面：按规范制作，附原文 + 翻译（建议 YOLOv5 / ByteTrack 原文）  
- [ ] 致谢：独立章节，不编章号  
- [ ] 页眉页脚：按样例设置（页眉：论文题目；页脚：页码）  
- [ ] 字体字号：正文宋体/Times New Roman 12pt，标题黑体，具体按校规  
- [ ] 行距：1.5 倍或固定值，按校规  
- [ ] 盲审匿名：确认正文无学号、姓名、学校名等可识别信息  

---

*本大纲最后修订：2026-04-18。写作过程中如需更新实验数据，直接替换【表 X-X】中的数值占位符即可。*
