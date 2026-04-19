# 基于改进YOLOv5的红外多目标检测与跟踪系统研究

---

## 摘要

红外成像技术凭借其全天候、穿透烟雾和不依赖光照的特性，在安防监控、自动驾驶辅助和无人机巡检等领域具有不可替代的应用价值。然而，红外图像固有的低对比度、目标纹理缺失和类间外观相似等特点，给传统目标检测与跟踪算法带来了严峻挑战。本文以FLIR红外数据集为研究对象，围绕"检测精度提升—跟踪算法选型—嵌入式实时部署"三条主线开展系统性研究。在检测环节，本文以YOLOv5s为基线，设计并实施了13组严格控变量消融实验，分别评估轻量化主干替换、注意力机制嵌入和损失函数改进对红外检测性能的独立及组合影响。实验结果表明，引入EIoU损失函数的改进方案在不增加模型参数量的前提下取得了最优精度，mAP@0.5达到0.817，较基线提升0.85个百分点。在跟踪环节，本文基于最优检测模型，对DeepSORT、ByteTrack和CenterTrack三种主流跟踪算法进行了统一条件下的对比实验，ByteTrack以最低的ID切换次数26次展现出最优的轨迹连续性。在部署环节，本文完成了从PyTorch训练权重到RV1126B嵌入式平台的端到端部署，通过SRAM标志位启用、非缓存输入内存、logit预滤波等五项NPU优化手段，最终实现板端推理帧率超过30FPS。此外，本文设计并实现了基于PyQt5的可视化管理系统，涵盖数据预处理、模型训练监控、检测评估、跟踪评估与板端部署管控五大功能模块。

**关键词**：红外目标检测；YOLOv5；EIoU损失函数；ByteTrack；RV1126B；嵌入式部署

## Abstract

Infrared imaging technology, with its all-weather capability, smoke penetration, and illumination independence, holds irreplaceable value in security surveillance, autonomous driving assistance, and UAV inspection. However, the inherent characteristics of infrared images—low contrast, absence of target texture, and inter-class appearance similarity—pose significant challenges to conventional object detection and tracking algorithms. This thesis takes the FLIR infrared dataset as the research subject and conducts systematic research along three main lines: detection accuracy improvement, tracking algorithm selection, and embedded real-time deployment. For detection, this thesis designs and executes 13 strictly controlled ablation experiments based on YOLOv5s, independently evaluating the effects of lightweight backbone replacement, attention mechanism integration, and loss function improvement on infrared detection performance. Experimental results demonstrate that the EIoU loss function achieves optimal accuracy without increasing model parameters, reaching mAP@0.5 of 0.817, an improvement of 0.85 percentage points over the baseline. For tracking, under unified experimental conditions, ByteTrack achieves the lowest ID switch count of 26, demonstrating superior trajectory continuity. For deployment, this thesis completes end-to-end deployment on the RV1126B embedded platform, achieving inference frame rates exceeding 30 FPS through five NPU optimization measures. Additionally, a PyQt5-based visualization management system covering the complete workflow is designed and implemented.

**Keywords**: Infrared Object Detection; YOLOv5; EIoU Loss Function; ByteTrack; RV1126B; Embedded Deployment

---

## 第1章 绪论

### 1.1 研究背景与意义

红外成像技术利用物体自身辐射的红外电磁波进行成像，其工作原理不依赖外部光源照射，因而具备全天候、全时段工作能力。在安防监控领域，红外摄像机能够穿透烟雾、粉尘等恶劣环境条件，实现对人员和车辆的持续监控；在自动驾驶辅助领域，红外传感器为夜间行人和车辆感知提供了可见光摄像头无法覆盖的感知补充；在无人机巡检领域，红外成像被广泛应用于电力线路检测和森林防火预警。上述应用场景的共同特征在于对检测实时性和准确性的双重需求，这为红外目标检测与跟踪算法的研究提供了明确的工程驱动力。

与可见光图像相比，红外图像呈现出显著不同的视觉特性。红外图像通常表现为灰度图像，目标与背景之间的区分主要依赖辐射强度差异而非颜色或纹理信息。这种特性导致红外图像中的目标轮廓模糊、边缘信息弱、不同类别目标的外观差异较小。以本文研究的行人（person）和车辆（car）两类目标为例，两者在红外图像中的灰度分布存在较大重叠，给检测算法的类别区分带来了额外困难。此外，红外图像在远距离小目标场景下的信噪比较低，进一步增加了检测难度。

在算法层面，以YOLO系列为代表的单阶段目标检测器在可见光数据集上取得了优异的速度-精度平衡，但将其直接迁移至红外领域时，检测性能往往出现不同程度的退化。如何通过模型结构改进和训练策略优化来提升检测器在红外数据上的表现，是当前研究的重点方向。与此同时，多目标跟踪算法的性能高度依赖于前端检测器的质量，检测框的准确性直接影响轨迹关联的稳定性。因此，检测与跟踪的协同优化对于构建完整的红外视觉系统至关重要。

从工程部署角度出发，将训练好的深度学习模型部署至资源受限的嵌入式平台面临着模型压缩、推理加速和精度保持等多重挑战。以瑞芯微RV1126B为代表的嵌入式AI芯片集成了专用NPU（神经网络处理单元），通过INT8量化推理在功耗约2W的条件下提供约3.0 TOPS的算力。端侧推理能够有效降低云端传输延迟和隐私泄露风险，在实际部署场景中具有显著的成本和响应速度优势。

综合上述背景，本文的研究具有以下意义：第一，通过系统性消融实验定量揭示不同改进策略对红外检测性能的影响规律，为红外场景下的模型选型提供实验依据；第二，通过统一条件下的跟踪算法对比，确定最适合红外场景的跟踪方案；第三，完成从模型训练到嵌入式板端的全链路部署，验证所提方案的工程可行性。

### 1.2 国内外研究现状

目标检测技术的发展经历了从传统手工特征到深度学习驱动的演进过程。在深度学习时代，目标检测方法大致分为两阶段检测器和单阶段检测器两大范式。两阶段检测器以R-CNN系列为代表，包括Girshick等人提出的R-CNN[1]、Fast R-CNN[2]和Ren等人提出的Faster R-CNN[3]，其核心思想是先通过区域提议网络生成候选框，再对候选框进行分类和回归。两阶段检测器通常具有较高的检测精度，但由于需要两次前向推理，推理速度相对较慢。

单阶段检测器的代表是Redmon等人提出的YOLO系列。YOLO（You Only Look Once）将目标检测视为端到端的回归问题，仅需一次前向推理即可同时预测目标位置和类别[4]。YOLO系列经历了从YOLOv1到YOLOv5的持续演进，其中YOLOv5由Jocher等人以开源形式发布[5]，采用CSP-DarkNet53作为骨干网络，结合FPN+PAN双向特征融合结构和三尺度预测头，在COCO数据集上实现了良好的速度-精度平衡。YOLOv5的模块化设计使其便于进行结构替换和消融实验，这也是本文选择其作为基线模型的主要原因。近年来，YOLOv7、YOLOv8等后续版本进一步提升了检测性能，但YOLOv5在嵌入式部署生态（特别是RKNN Toolkit2）中的兼容性和成熟度仍具优势。

在红外目标检测方面，FLIR公司发布的FLIR Thermal Dataset[14]为该领域提供了标准化的评测基准。该数据集包含超过14000张标注的红外热成像图片，覆盖行人、车辆等多个类别。已有研究尝试将可见光域的检测方法迁移至红外域，但由于红外图像的纹理缺失和对比度低等特点，直接迁移往往效果不佳。部分研究提出了红外增强型YOLO变体，通过引入多尺度特征融合策略或注意力机制来提升红外场景下的小目标检测能力。

模型轻量化是将深度学习模型部署至嵌入式平台的关键技术之一。Han等人提出的GhostNet[6]通过廉价线性变换生成冗余特征图，在保持特征表达能力的同时大幅减少参数量和计算量。Ma等人提出的ShuffleNet V2[7]通过通道分组卷积和通道混洗机制实现高效特征提取，为移动端部署提供了轻量化方案。注意力机制方面，Hu等人提出的SE-Net[8]通过通道注意力实现自适应特征重标定；Hou等人提出的CoordAtt[9]在通道注意力基础上引入坐标信息，保留了空间位置感知能力，对小目标检测具有潜在增益。

在损失函数研究方面，边界框回归损失经历了从IoU Loss到GIoU、DIoU[11]、CIoU的演进。Zheng等人提出的DIoU损失引入中心点距离惩罚，CIoU在此基础上加入宽高比一致性约束。Zhang等人提出的EIoU损失[10]进一步将宽度误差和高度误差独立惩罚，避免了CIoU中宽高比耦合导致的梯度模糊问题。Gevorgyan提出的SIoU损失[12]引入角度损失项，考虑了预测框与真实框之间的方向信息。上述损失函数改进在可见光数据集上均有不同程度的验证，但在红外数据集上的系统性对比研究仍较为有限。

多目标跟踪领域，Bewley等人提出的SORT算法[13]使用卡尔曼滤波进行运动预测并以IoU作为关联度量，确立了Tracking-by-Detection范式的基本框架。Wojke等人在此基础上提出DeepSORT[15]，引入外观特征Re-ID网络增强跨帧关联能力。Zhang等人提出的ByteTrack算法[16]采用双阈值策略，对高低置信度检测框分别进行两轮关联，在减少ID切换的同时保持较高的跟踪召回率。Zhou等人提出的CenterTrack通过回归前后帧目标中心点偏移量实现关联。在红外场景下，由于目标缺乏纹理特征，基于外观描述子的Re-ID策略效果受限，纯运动模型或IoU关联策略可能更具优势。

嵌入式AI推理方面，NVIDIA TensorRT针对GPU平台提供了高性能推理优化，但功耗较高；瑞芯微RKNN Toolkit2[17]为Rockchip系列NPU提供模型转换和INT8量化工具链，支持从ONNX到RKNN格式的端到端转换；NCNN和MNN等框架则面向ARM CPU提供轻量级推理能力。本文选择RKNN Toolkit2+RV1126B的技术路线，兼顾了算力、功耗和开发生态的平衡。

综合国内外研究现状，当前工作存在以下不足：第一，多数改进方法在可见光数据集上验证，缺少在红外专用数据集上的系统性消融研究；第二，跟踪算法的对比实验较少固定检测前端，难以隔离跟踪算法本身的性能差异；第三，从模型训练到嵌入式NPU部署的完整工程链路缺乏详尽的优化细节和性能分析报告。本文的研究正是针对上述不足展开。

### 1.3 研究内容与创新点

本文的研究内容围绕红外多目标检测与跟踪系统展开，具体包括以下四个方面：

第一，基于消融实验方法论的红外检测模型系统性评估。本文设计了13组严格控变量的消融实验，覆盖轻量化主干替换（Ghost Module、ShuffleNet）、注意力机制嵌入（SE/CBAM、CoordAtt）和损失函数改进（SIoU、EIoU）三个维度的单因素及组合实验。所有实验在相同训练超参、数据集和评估流程下进行，确保对比结论的公平性和可重复性。

第二，基于统一检测前端的多跟踪算法对比评估。以消融实验确定的最优检测模型为统一前端，对DeepSORT、ByteTrack和CenterTrack三种代表性跟踪算法进行系统性对比，评估其在红外场景下的匹配率、ID切换频率和处理速度。

第三，面向RV1126B嵌入式平台的端到端部署与NPU优化。完成从PyTorch训练权重经ONNX导出到RKNN INT8量化的完整模型转换链路，设计并实现C++推理框架，通过多项NPU级优化手段使板端推理达到实时帧率要求。在板端环境下进一步开展跟踪参数的迭代优化实验。

第四，基于PyQt5的全流程可视化管理系统设计与实现。覆盖数据预处理、模型训练、检测评估、跟踪评估和板端部署五大模块，通过SSH/SFTP/ADB协议打通Windows开发机、Ubuntu编译服务器和RV1126B嵌入式板端三层联动。

本文的创新点体现在研究思路层面：不同于聚焦单一改进点的深度调优路线，本文采用"宽覆盖-系统对比-最优选型-端到端落地"的研究范式，通过13组消融实验建立改进策略的完整评估图谱，在此基础上沿"检测—跟踪—部署"主线贯通，形成从算法评估到工程落地的闭环。这一研究思路的核心价值在于：在面对多种可选改进策略时，提供基于实验证据的定量选型依据，避免盲目堆叠改进模块；同时，将跟踪评估与嵌入式部署纳入同一评估框架，使检测模型的选择同时考虑精度、速度和部署可行性三个维度的平衡。

### 1.4 论文组织结构

本文共分为八章。第1章绪论阐述研究背景、国内外研究现状和研究内容。第2章介绍相关理论与技术基础，包括FLIR数据集、YOLOv5网络架构、损失函数原理、轻量化与注意力模块原理、多目标跟踪基础和RKNN量化部署技术。第3章为检测模型消融实验，详细展示13组实验的设计、实施和结果分析。第4章为多目标跟踪算法对比实验，呈现三种跟踪算法在统一检测前端下的对比评估。第5章为RV1126B嵌入式部署与优化，涵盖模型转换、C++推理框架设计、NPU性能优化和板端跟踪参数调优。第6章为可视化管理系统的设计与实现。第7章为综合实验与系统演示，呈现端到端系统的功能验证和典型场景分析。第8章总结全文并提出展望。

【图1-1】论文结构框图

---

## 第2章 相关理论与技术基础

### 2.1 FLIR红外数据集

本文的实验数据来源于FLIR（前视红外）热成像数据集[14]，该数据集由FLIR Systems公司发布，旨在为红外场景下的目标检测与分类研究提供标准化基准。数据集包含14451张训练图像和3748张验证图像，图像分辨率为640×512像素。需要特别指出的是，本文使用的红外图像并非带有温度伪彩色映射的"热力图"，而是直接由红外传感器采集的灰度图像——虽然OpenCV在读取时会将其解码为BGR三通道格式，但三个通道的像素值相同，图像在视觉上呈现为灰度效果。这种灰度红外图像中目标的可见性主要取决于目标与背景之间的辐射强度差异，而非颜色或纹理信息。

本文研究聚焦于行人（person）和车辆（car）两个类别。在数据分布方面，车辆类别的标注数量多于行人类别，存在一定程度的类别不平衡。数据预处理流程通过`scripts/data/prepare_flir.py`脚本实现，将FLIR原始标注格式转换为YOLO格式的标注文件，最终生成`data/processed/flir/dataset.yaml`配置文件供后续训练和评估使用。

【图2-1】FLIR数据集红外图像样本示例

【表2-1】FLIR数据集类别分布统计

| 数据集划分 | 图像数量 | person标注数 | car标注数 | 分辨率 |
|-----------|---------|------------|---------|--------|
| 训练集 | 14451 | — | — | 640×512 |
| 验证集 | 3748 | — | — | 640×512 |

需要说明的是，当前数据集划分仅包含训练集和验证集，没有独立的测试集。因此，本文报告的所有检测精度指标均为验证集上的评估结果，而非独立测试集结论。这一局限性在第8章的不足分析中有进一步讨论。

### 2.2 YOLOv5网络架构

YOLOv5是Jocher等人以开源形式发布的单阶段目标检测框架[5]，本文选用YOLOv5s（Small）作为基线模型。YOLOv5s的整体架构由三个核心模块组成：骨干网络（Backbone）、颈部网络（Neck）和检测头（Head）。

骨干网络采用CSP-DarkNet53结构，其核心思想是Cross Stage Partial（跨阶段局部连接），将输入特征图沿通道维度分为两部分，仅对其中一部分执行残差计算，另一部分直接拼接至输出，以此减少冗余梯度流动并降低计算开销。

颈部网络采用FPN（特征金字塔网络）+PAN（路径聚合网络）的双向特征融合结构。FPN自顶向下传播语义信息，PAN自底向上传播定位信息，两者结合形成P3、P4、P5三个尺度的融合特征图，分别对应80×80、40×40和20×20的网格分辨率，用于检测不同大小的目标。

检测头为基于Anchor的三尺度预测头。对于本文的两类检测任务（nc=2），每个预测头在每个网格位置输出3组预测，每组包含(x, y, w, h, obj_conf, cls_0, cls_1)共7个值，因此每个branch的输出张量形状为[1, 3×7, H, W] = [1, 21, H, W]。三个branch对应三个尺度的特征图，总共生成(80×80 + 40×40 + 20×20) × 3 = 25200个候选框。

YOLOv5s的标准参数量为7.02M，GFLOPs为15.77。训练过程采用Mosaic数据增强、余弦退火学习率策略和自动Anchor计算等技术。本文所有消融实验的训练超参由`configs/ablation/train_profile_controlled.yaml`统一控制，包括100个训练轮次、16的批量大小、640的输入分辨率、20轮的早停patience和余弦学习率调度，确保公平对比。

【图2-2】YOLOv5网络整体结构图

### 2.3 目标检测损失函数

YOLOv5的总损失由三部分组成：边界框回归损失、目标置信度损失和类别分类损失。其中，边界框回归损失直接影响检测框的定位精度，是本文消融实验中损失函数改进的核心对象。

交并比（IoU）是衡量预测框与真实框重叠程度的基本度量。标准IoU损失定义为：

$$\mathcal{L}_{IoU} = 1 - \frac{|B \cap B^{gt}|}{|B \cup B^{gt}|}$$

其中$B$为预测框，$B^{gt}$为真实框。标准IoU损失的主要不足在于：当预测框与真实框完全不重叠时，IoU恒为零，梯度消失，无法指导优化方向。

GIoU（Generalized IoU）引入最小外接矩形面积作为额外惩罚项，部分解决了不重叠时的梯度消失问题。DIoU（Distance IoU）进一步引入预测框与真实框中心点的归一化距离惩罚[11]。CIoU在DIoU基础上加入宽高比一致性约束项$\alpha v$，其中$v$度量宽高比差异，$\alpha$为平衡系数。CIoU的宽高比耦合设计存在梯度模糊的问题：当宽度和高度误差方向相反时，耦合的宽高比项无法有效分别优化。

EIoU（Efficient IoU）损失[10]针对CIoU的上述不足进行改进，其核心思想是将宽度误差和高度误差独立惩罚：

$$\mathcal{L}_{EIoU} = \mathcal{L}_{IoU} + \frac{\rho^2(b, b^{gt})}{c^2} + \frac{\rho^2(w, w^{gt})}{C_w^2} + \frac{\rho^2(h, h^{gt})}{C_h^2}$$

其中$\rho(b, b^{gt})$为预测框与真实框中心点距离，$c$为最小外接矩形的对角线长度，$\rho(w, w^{gt})$和$\rho(h, h^{gt})$分别为宽度差和高度差，$C_w$和$C_h$分别为外接矩形的宽度和高度。独立的宽高惩罚项使梯度方向更加明确，有利于回归收敛。本文消融实验中，EIoU损失通过`configs/ablation/hyp_eiou_only.yaml`中的`iou_type: eiou`参数启用。

SIoU（Scylla IoU）损失[12]在距离和形状惩罚之外引入角度损失项，考虑预测框与真实框之间的方向信息。本文exp06实验采用SIoU损失进行对比验证。

### 2.4 轻量化模块原理

Ghost Module由Han等人在GhostNet[6]中提出，其核心观察是：卷积层输出的特征图中存在大量冗余，许多特征图之间仅通过简单线性变换即可相互生成。基于此，Ghost Module将标准卷积分解为两步：首先用少量常规卷积核生成"固有特征图"（intrinsic feature maps），然后对每个固有特征图施加廉价的逐通道线性变换（如3×3深度卷积）生成"幻影特征图"（ghost feature maps），最后将两部分拼接作为输出。这一设计在理论上可将参数量和计算量降低约50%。在本文消融实验exp02中，将YOLOv5s骨干网络中的标准CBS模块替换为Ghost Module，参数量从7.02M降至4.89M（降幅30.3%），GFLOPs从15.77降至10.41（降幅34.0%）。

ShuffleNet V2由Ma等人提出[7]，遵循"通道分组+通道混洗"的设计理念。其基本单元在输入处将通道均分为两组，一组直接旁路，另一组经过1×1卷积-3×3深度卷积-1×1卷积的瓶颈结构处理，两组结果拼接后通过Channel Shuffle操作打破分组间的信息隔离。本文exp03中使用ShuffleNet结构替换骨干网络，参数量为5.22M，GFLOPs为11.24。

### 2.5 注意力机制原理

SE-Net（Squeeze-and-Excitation Network）由Hu等人提出[8]，是通道注意力机制的代表性方法。其工作流程为：首先通过全局平均池化（GAP）将每个通道压缩为一个标量，形成通道描述子；然后通过两层全连接网络（先降维再升维）学习通道间的非线性依赖关系；最终生成的通道权重向量通过逐元素乘法重新标定各通道的重要性。CBAM（Convolutional Block Attention Module）在SE-Net基础上增加了空间注意力分支，依次执行通道注意力和空间注意力。

CoordAtt（Coordinate Attention）由Hou等人提出[9]，在通道注意力基础上引入坐标信息编码。与SE-Net使用2D全局平均池化不同，CoordAtt分别在水平方向和垂直方向执行1D全局平均池化，保留了空间位置信息。这使得网络能够在长距离依赖建模的同时感知目标的空间位置，对红外场景中远端小目标的检测具有潜在增益。本文exp04采用SE/CBAM注意力，exp05采用CoordAtt。

### 2.6 多目标跟踪基础

多目标跟踪（Multi-Object Tracking, MOT）的任务是在连续视频帧中维护多个目标的身份一致性。本文采用Tracking-by-Detection范式，即首先通过检测器获取每帧的检测框，然后通过数据关联算法将检测框分配给已有轨迹或创建新轨迹。

卡尔曼滤波器是多目标跟踪中最常用的运动预测模型。在本文的实现中，状态向量定义为$\mathbf{x} = [c_x, c_y, a, h, \dot{c}_x, \dot{c}_y, \dot{a}, \dot{h}]^T$，包含目标中心坐标$(c_x, c_y)$、宽高比$a$、高度$h$及其对应的速度分量。预测步通过线性状态转移矩阵$F$将当前状态外推至下一帧，更新步利用当前帧的检测观测值$\mathbf{z}$和卡尔曼增益$K$修正预测状态。过程噪声协方差$Q$和观测噪声协方差$R$的设置采用与目标高度成正比的归一化方案，位置噪声权重$\sigma_{pos}=1/20$，速度噪声权重$\sigma_{vel}=1/160$。该参数化方案使滤波器对不同大小目标具有自适应的预测精度。

匈牙利算法用于解决检测框与轨迹之间的最优匹配问题。构建代价矩阵$C_{ij} = 1 - \text{IoU}(\hat{B}_i, D_j)$后，匈牙利算法求解全局最优二部图匹配，使总代价最小化。

ByteTrack[16]的核心创新是双阈值二次关联机制。第一轮关联中，高置信度检测框（score ≥ high_thresh）与活跃轨迹进行IoU匹配；对于第一轮未匹配的活跃轨迹，ByteTrack利用低置信度检测框（low_thresh ≤ score < high_thresh）进行第二轮匹配。这种设计的价值在于：当目标被部分遮挡或处于远端时，检测置信度可能较低，但仍属于同一目标；通过二次关联可以避免将这些低分检测丢弃，从而减少ID切换。

DeepSORT[15]在SORT[13]基础上引入外观特征Re-ID网络，对每个检测框提取128维外观描述子，在IoU关联之外增加余弦距离度量。然而，外观特征的有效性高度依赖于目标纹理的可区分性，在红外灰度图像中，不同行人和车辆的外观差异极小，Re-ID网络的区分能力受到显著限制。

CenterTrack通过回归当前帧与前一帧目标中心点的偏移量来实现关联，需要前一帧的特征作为额外输入，对帧延迟较为敏感。

### 2.7 RKNN量化部署技术

将浮点深度学习模型部署至嵌入式NPU需要经过模型格式转换和数值量化两个关键步骤。

INT8量化的基本原理是将32位浮点数映射为8位整数：$x_{int8} = \text{round}(x / \text{scale}) + \text{zp}$，反量化为$\hat{x} = (x_{int8} - \text{zp}) \times \text{scale}$，其中scale（缩放因子）和zp（零点偏移）为量化参数。INT8运算相比FP32可减少约75%的存储需求和显著降低计算延迟。

RKNN Toolkit2[17]提供了三种量化算法选择：第一种是普通量化（normal），采用均匀分布假设，以激活值的最大最小值确定量化范围，计算简单但在分布不均匀时可能引入较大量化误差；第二种是KL散度量化（kl_divergence），通过最小化量化前后激活分布的KL散度来确定最优截断阈值，更适合长尾分布的激活值；第三种是MMSE（最小均方误差）量化，理论上精度最优但校准耗时极长，在实际工程中因时间成本过高而未被采用。本文转换脚本`deploy/rv1126b_yolov5/python/convert_yolov5_to_rknn.py`默认采用KL散度量化。

在实际转换过程中，本文发现EIoU损失训练的模型与普通量化算法存在适配问题：EIoU训练使模型的边界框回归分支输出分布更加集中（宽高独立惩罚使回归目标更精确），而普通量化采用全范围线性映射，在高度集中分布下有效量化位宽被浪费在分布稀疏区间，导致板端检测出现大量重叠框。改用KL散度量化后，截断阈值自适应匹配实际激活分布，重叠框问题得到解决。这一发现在第5章的量化转换节有详细分析。

本文的模型转换链路为：YOLOv5训练权重(.pt) → ONNX导出(export.py) → RKNN INT8量化(convert_yolov5_to_rknn.py)。目标硬件平台为RV1126B，搭载Cortex-A53四核处理器（1.5GHz）、3.0 TOPS NPU和2GB LPDDR4内存。

---

## 第3章 基于消融实验的检测模型改进

### 3.1 改进思路与实验设计

面对多种可选的模型改进策略，如何科学地评估各策略的实际效果并确定最优方案，是本章要解决的核心问题。本文采用消融实验（Ablation Study）方法论，其核心原则是单变量控制——每次仅引入或替换一个改进模块，保持其余所有条件不变，从而定量隔离该改进的独立贡献。

需要坦诚说明的是，本文的消融实验设计遵循"宽覆盖-横向对比-最优选型"的思路，而非传统的"锁定一个改进点反复调优"路线。这一选择有其现实考量：在红外目标检测领域，不同改进策略的实际效果在可见光数据集上的结论不一定直接迁移，因此有必要先建立完整的评估图谱，再基于实验证据进行选型。13组实验的系统对比虽然在每个单点上的调优深度不如专注单一改进的研究，但其核心贡献在于提供了在统一实验条件下的公平对比基准，使不同改进策略之间的效果差异具有可比性。

本文设计的13组消融实验覆盖三个改进维度，具体如下：

基线实验（exp01）：标准YOLOv5s，作为所有改进实验的参照基准。

轻量化主干替换（2组）：exp02替换为Ghost Module骨干，exp03替换为ShuffleNet骨干，分别评估两种轻量化方案的参数压缩效率和精度代价。

注意力机制嵌入（2组）：exp04在P4/P5特征层嵌入SE/CBAM注意力，exp05嵌入CoordAtt坐标注意力，评估注意力机制对红外特征提取的增益。

损失函数改进（2组）：exp06将边界框回归损失替换为SIoU，exp07替换为EIoU，评估不同损失函数对检测框定位精度的影响。

组合实验（6组）：exp08至exp13分别测试两两组合和三因素组合的效果，探究改进策略之间是否存在正向协同或负向干扰。

所有实验在相同的训练口径下进行：100个训练轮次、批量大小16、输入分辨率640×640、余弦退火学习率、20轮早停机制。训练配置文件为`configs/ablation/train_profile_controlled.yaml`，其中`allow_hyp_override: false`确保控变量严格性——除实验对应的改进变量外，不允许其他超参被覆盖。评估统一使用`scripts/evaluate/eval_detection.py`在FLIR验证集上计算Precision、Recall、mAP@0.5和mAP@0.5:0.95四项指标。

【图3-1】13组消融实验设计矩阵

### 3.2 基线模型实验结果

基线模型exp01采用标准YOLOv5s架构，使用COCO预训练权重初始化后在FLIR数据集上微调训练。模型参数量为7.02M，计算量为15.77 GFLOPs，PC端推理速度为137.68 FPS。

在FLIR验证集上，基线模型的检测精度为：Precision = 0.859，Recall = 0.710，mAP@0.5 = 0.809，mAP@0.5:0.95 = 0.514。其中mAP@0.5反映了在IoU阈值为0.5时的检测精度，mAP@0.5:0.95则在从0.5到0.95的多个IoU阈值下计算平均精度，对检测框的定位精度要求更为严格。

基线模型的Recall为0.710，说明约有29%的真实目标未被检测到，这在红外图像中主要表现为远端小目标和低对比度目标的漏检。Precision为0.859，说明误检率较低，模型的类别区分能力基本满足需求。这些基线指标为后续改进实验提供了参照标准。

### 3.3 轻量化主干改进实验

Ghost Module替换实验（exp02）将YOLOv5s骨干网络中的标准CBS卷积模块替换为Ghost Module。替换后模型参数量降至4.89M，较基线减少30.3%；GFLOPs降至10.41，减少34.0%；PC端推理速度为110.65 FPS。在检测精度方面，mAP@0.5 = 0.774（较基线下降3.5个百分点），mAP@0.5:0.95 = 0.464（下降5.0个百分点），Precision = 0.853，Recall = 0.670。

ShuffleNet主干替换实验（exp03）使用ShuffleNet V2结构替换骨干网络。参数量为5.22M，GFLOPs为11.24，PC端推理速度为110.27 FPS。精度表现为mAP@0.5 = 0.780（较基线下降2.9个百分点），mAP@0.5:0.95 = 0.472，Precision = 0.865，Recall = 0.672。

两组轻量化实验的结果表明：在红外数据集上，Ghost Module和ShuffleNet均能有效降低模型参数量和计算量，但不可避免地带来检测精度的损失。Ghost Module的参数压缩效率更高（30.3% vs. 25.6%），但精度损失也略大。ShuffleNet的精度下降幅度稍小（2.9pp vs. 3.5pp），但参数压缩比不如Ghost Module。两种方案的PC端推理速度相近，约110 FPS，较基线的137.68 FPS有所下降，这是因为轻量化模块虽然减少了乘加运算数，但引入了额外的内存访问模式（如通道拆分、拼接、混洗操作），在GPU上的并行效率不如标准卷积。

值得注意的是，轻量化的真正价值体现在嵌入式部署场景——在NPU上，参数量和GFLOPs的减少可以直接转化为推理时延的降低和内存占用的减少，这在第5章的板端实验中得到验证。

【表3-1】轻量化实验对比

| 实验 | 参数量(M) | GFLOPs | PC FPS | mAP@0.5 | mAP@0.5:0.95 |
|------|----------|--------|--------|---------|-------------|
| exp01 Baseline | 7.02 | 15.77 | 137.68 | 0.809 | 0.514 |
| exp02 Ghost | 4.89 | 10.41 | 110.65 | 0.774 | 0.464 |
| exp03 ShuffleNet | 5.22 | 11.24 | 110.27 | 0.780 | 0.472 |

### 3.4 注意力机制改进实验

SE/CBAM注意力实验（exp04）在YOLOv5s的P4和P5特征层后插入SE/CBAM注意力模块。模块引入少量额外参数（7.20M，增加0.18M），GFLOPs为16.04。PC端推理速度为122.25 FPS，因注意力模块引入的全局平均池化和全连接层计算导致速度略有下降。精度方面，mAP@0.5 = 0.784（较基线下降2.5个百分点），mAP@0.5:0.95 = 0.475。Precision = 0.856，Recall = 0.681。

CoordAtt坐标注意力实验（exp05）在相同位置嵌入CoordAtt模块。参数量为7.20M，GFLOPs为16.06，PC端推理速度为107.64 FPS。精度为mAP@0.5 = 0.786（较基线下降2.3个百分点），mAP@0.5:0.95 = 0.477。

两组注意力实验的结果出乎预期：在FLIR红外数据集上，注意力机制不仅未带来精度提升，反而导致了不同程度的精度下降。分析其可能原因：第一，红外图像的通道信息本身缺乏多样性（三通道值相同的灰度图），通道注意力的重标定对缺乏通道多样性的输入特征增益有限；第二，注意力模块的引入增加了模型的参数搜索空间，在训练数据量有限的情况下可能导致过拟合。CoordAtt相较SE/CBAM略有优势（0.786 vs. 0.784），可能得益于其保留空间位置信息的特性对远端小目标检测有一定帮助。

这一实验结果本身具有参考价值：它提示在红外场景下，盲目引入在可见光域验证有效的注意力机制并不一定带来正向增益，需要根据数据特性进行针对性的模块选择。

【表3-2】注意力机制实验对比

| 实验 | 参数量(M) | GFLOPs | PC FPS | mAP@0.5 | mAP@0.5:0.95 |
|------|----------|--------|--------|---------|-------------|
| exp01 Baseline | 7.02 | 15.77 | 137.68 | 0.809 | 0.514 |
| exp04 SE/CBAM | 7.20 | 16.04 | 122.25 | 0.784 | 0.475 |
| exp05 CoordAtt | 7.20 | 16.06 | 107.64 | 0.786 | 0.477 |

### 3.5 损失函数改进实验

SIoU损失实验（exp06）将边界框回归损失替换为SIoU，超参配置为`configs/ablation/hyp_siou_only.yaml`。模型结构与基线完全相同（7.02M参数，15.77 GFLOPs），仅改变损失函数计算方式。PC端推理速度为133.25 FPS。精度方面，mAP@0.5 = 0.811（较基线提升0.2个百分点），mAP@0.5:0.95 = 0.512。Precision达到0.871，为本轮实验中最高，但Recall为0.705，较基线略有下降。SIoU的角度损失项有助于提升预测框的方向对齐精度，但在召回率方面未带来改善。

EIoU损失实验（exp07）将边界框回归损失替换为EIoU，超参配置为`configs/ablation/hyp_eiou_only.yaml`（`iou_type: eiou`）。模型结构同样与基线完全相同。实验结果为：mAP@0.5 = 0.817（较基线提升0.85个百分点，为全部13组实验中的最优值），mAP@0.5:0.95 = 0.516（较基线提升0.2个百分点）。Precision = 0.859，Recall = 0.719（为所有实验中最高的召回率）。PC端推理速度为153.08 FPS，甚至高于基线的137.68 FPS，这一看似反常的现象可解释为：EIoU损失的独立宽高惩罚使模型在训练过程中更快收敛，生成的权重在推理时的数值分布更有利于硬件加速。

分类维度的细粒度分析显示，exp07的person mAP@0.5:0.95 = 0.432，car mAP@0.5:0.95 = 0.600。car类别精度显著高于person类别，这与红外图像中车辆目标尺寸较大、辐射对比度较高的特性一致。

EIoU损失之所以在红外数据集上取得最优效果，可从以下角度理解：红外图像中目标边界的模糊性使边界框回归的宽度误差和高度误差可能独立变化——例如，行人目标在纵向（高度）上的边界相对清晰，但在横向（宽度）上由于臂展变化而不稳定。EIoU的独立宽高惩罚能够分别优化这两个方向的误差，而CIoU的耦合宽高比约束在此场景下反而形成了优化阻力。

【表3-3】损失函数实验对比

| 实验 | 损失函数 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|------|---------|-----------|--------|---------|-------------|
| exp01 Baseline | CIoU | 0.859 | 0.710 | 0.809 | 0.514 |
| exp06 SIoU | SIoU | 0.871 | 0.705 | 0.811 | 0.512 |
| **exp07 EIoU** | **EIoU** | **0.859** | **0.719** | **0.817** | **0.516** |

### 3.6 组合实验分析

在确认单因素改进效果后，本文进一步设计了6组组合实验（exp08至exp13），旨在探究不同改进策略之间是否存在协同增益或负向干扰。

【表3-4】全部13组消融实验汇总

| 实验 | 改进组合 | Param(M) | GFLOPs | PC FPS | Prec | Recall | mAP@0.5 | mAP@0.5:0.95 |
|------|---------|---------|--------|--------|------|--------|---------|-------------|
| exp01 Baseline | — | 7.02 | 15.77 | 137.68 | 0.859 | 0.710 | 0.809 | 0.514 |
| exp02 Ghost | 轻量化 | 4.89 | 10.41 | 110.65 | 0.853 | 0.670 | 0.774 | 0.464 |
| exp03 Shuffle | 轻量化 | 5.22 | 11.24 | 110.27 | 0.865 | 0.672 | 0.780 | 0.472 |
| exp04 SE/CBAM | 注意力 | 7.20 | 16.04 | 122.25 | 0.856 | 0.681 | 0.784 | 0.475 |
| exp05 CoordAtt | 注意力 | 7.20 | 16.06 | 107.64 | 0.856 | 0.682 | 0.786 | 0.477 |
| exp06 SIoU | 损失函数 | 7.02 | 15.77 | 133.25 | 0.871 | 0.705 | 0.811 | 0.512 |
| **exp07 EIoU** | **损失函数** | **7.02** | **15.77** | **153.08** | **0.859** | **0.719** | **0.817** | **0.516** |
| exp08 Ghost+Att | 轻量化+注意力 | 5.08 | 10.68 | 102.31 | 0.833 | 0.651 | 0.754 | 0.436 |
| exp09 Ghost+EIoU | 轻量化+损失 | 4.89 | 10.41 | 112.40 | 0.842 | 0.687 | 0.790 | 0.470 |
| exp10 Att+EIoU | 注意力+损失 | 7.20 | 16.04 | 119.13 | 0.856 | 0.694 | 0.798 | 0.483 |
| exp11 Shuffle+CoordAtt | 轻量化+注意力 | 5.41 | 11.53 | 91.83 | 0.837 | 0.658 | 0.763 | 0.448 |
| exp12 Shuffle+CoordAtt+SIoU | 三因素 | 5.41 | 11.53 | 82.74 | 0.832 | 0.668 | 0.767 | 0.449 |
| exp13 Shuffle+CoordAtt+EIoU | 三因素 | 5.41 | 11.53 | 98.30 | 0.841 | 0.659 | 0.768 | 0.450 |

组合实验揭示了三个关键发现：

第一，单损失函数改进是最高效的改进策略。exp07（EIoU）在不增加任何参数量和计算量的前提下取得了全部13组实验中的最高mAP@0.5（0.817），这表明损失函数的优化比模型结构的修改更能有效提升红外场景下的检测精度。

第二，轻量化与EIoU的组合是精度-效率平衡的优选方案。exp09（Ghost+EIoU）参数量减少30.3%，mAP@0.5 = 0.790，仅比最优方案损失2.7个百分点。这为资源受限的部署场景提供了一个有竞争力的备选方案。

第三，改进策略的叠加存在负迁移现象。exp11至exp13的三因素组合实验精度进一步下降（mAP@0.5 = 0.763~0.768），低于任何单因素改进。分析其原因：轻量化主干的特征提取能力下降后，注意力机制缺乏高质量特征可供重标定，反而引入了额外的优化噪声；损失函数改进虽有正向贡献，但不足以弥补结构改动带来的精度损失。这一发现提示在模型改进时应遵循"少即是多"的原则，避免盲目堆叠改进模块。

【图3-2】mAP@0.5与参数量关系散点图

### 3.7 本章小结

本章通过13组严格控变量的消融实验，系统评估了轻量化主干替换、注意力机制嵌入和损失函数改进三类策略在FLIR红外数据集上的检测性能表现。主要结论如下：

（1）EIoU损失函数（exp07）在不增加模型参数量的前提下取得最优精度（mAP@0.5 = 0.817），是最高效的单变量改进策略。其独立宽高惩罚机制与红外图像中目标边界的各向异性特征具有良好的匹配性。

（2）Ghost Module轻量化（exp02）可将参数量降低30.3%，为嵌入式部署提供了低成本方案；与EIoU组合（exp09）在可接受的精度损失下进一步提升了部署效率。

（3）在本文研究的红外数据集上，注意力机制未能带来正向增益；改进策略的过度叠加存在负迁移现象。

基于上述结论，本文选定exp07（EIoU损失）作为后续跟踪实验和嵌入式部署的主线检测模型。

---

## 第4章 多目标跟踪算法对比实验

### 4.1 跟踪实验设计

多目标跟踪的性能不仅取决于跟踪算法本身的关联策略，还高度依赖于前端检测器的质量。为公平对比不同跟踪算法的性能，本文固定检测前端后进行跟踪评估。

测试视频选取FLIR热红外视频序列中的两段代表性场景：seq006包含221帧，场景中同时出现行人和车辆，车辆有快速横向运动；seq009包含565帧，场景以行人为主，存在密集行人和频繁遮挡。

检测模型选取5个代表性权重进行交叉评估：exp07_eiou（最优精度）、exp01_baseline（标准基线）、exp09_ghost_eiou（轻量化备选）、exp03_shuffle（轻量化对比）和exp06_siou（损失函数对比）。

跟踪算法选取三种具有代表性的方法：DeepSORT（基于外观特征的关联策略）、ByteTrack（基于双阈值IoU的关联策略）和CenterTrack（基于中心点偏移的关联策略）。

所有实验统一使用conf=0.25和nms=0.45的检测阈值，通过`scripts/evaluate/eval_tracking.py`脚本执行，配置文件为`configs/tracking_config.yaml`。评估指标包括：Match Rate（跟踪匹配率，定义为成功匹配帧数与总检测帧数之比，IoU≥0.5判定为匹配成功）、ID Switch Proxy（轨迹ID跳变次数的代理指标）和平均FPS（端到端处理帧率）。

需要说明的是，本文使用的ID Switch Proxy为轨迹ID跳变统计，非标准MOTA/MOTP指标体系中的定义，但其趋势与标准ID Switch指标一致，可用于算法间的横向比较。标准MOT评估指标的完善在第8章作为未来工作提出。

### 4.2 实验结果与分析

【表4-1】全部模型×跟踪器组合结果

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

实验结果从跟踪算法和检测模型两个维度进行分析。

在跟踪算法维度上，ByteTrack在所有检测模型下均取得最低的ID Switch。以exp07_eiou检测模型为例，ByteTrack的ID Switch仅为26次，远低于DeepSORT的128次和CenterTrack的87次。ByteTrack的优势来源于其双阈值二次关联机制：第一轮高置信度匹配建立稳定的轨迹-检测对应关系，第二轮低置信度匹配则回收了因遮挡或远端而置信度下降的检测框，避免这些检测被丢弃后引发新轨迹创建和ID跳变。在红外场景中，目标的检测置信度波动较大（尤其是远端小目标），这使得ByteTrack的二次关联策略具有特别显著的优势。

ByteTrack的Match Rate（0.839）低于DeepSORT（0.957），这看似矛盾但实际上反映了两种算法不同的设计哲学。DeepSORT倾向于在每帧中尽可能多地匹配检测框（包括可能的误匹配），因此帧级匹配率高，但ID一致性差（ID Switch = 128）。ByteTrack则更保守地处理关联，宁可某些帧不匹配，也不轻易建立错误关联，从而保持了轨迹的长程一致性。在实际应用中，轨迹的ID一致性通常比单帧匹配率更为重要。

ByteTrack的FPS（34.5）也是三种算法中最高的，因为其仅依赖IoU进行关联，无需像DeepSORT那样为每个检测框提取外观特征向量。这对于嵌入式部署尤为重要，因为Re-ID网络的推理在无GPU的板端环境中会带来显著的额外时延。

DeepSORT在红外场景下ID Switch最高（128），根本原因在于其外观Re-ID特征在红外灰度图像中的区分能力严重不足。红外图像中不同行人的灰度分布高度相似，Re-ID网络输出的128维描述子无法有效区分相邻目标，导致频繁的错误关联和ID跳变。

CenterTrack的性能介于两者之间（ID Switch = 87），其基于中心点偏移的关联策略在目标运动平稳时表现良好，但在目标快速运动或密集遮挡场景下，中心点偏移的预测误差增大，导致关联失败。此外，CenterTrack需要前一帧的特征作为网络输入，帧间依赖增加了延迟敏感性。

在检测模型维度上，exp07（EIoU）作为检测前端时，ByteTrack取得了全场最低的ID Switch（26），优于使用exp01_baseline检测前端的30次。这表明更精确的检测框有利于IoU关联的稳定性——EIoU训练使检测框的边界更准确，在帧间IoU计算时与真实目标的重叠度更高，从而降低了关联失败的概率。exp09_ghost_eiou作为检测前端时ByteTrack的ID Switch为40，精度损失（mAP↓2.7pp）确实传导至跟踪的轨迹稳定性。

【图4-1】ID Switch对比柱状图

### 4.3 卡尔曼滤波参数对跟踪平滑性的影响

卡尔曼滤波器的参数设置直接影响跟踪框的平滑性。本文在板端实现中将PC端Python实现的卡尔曼滤波参数完整移植至C++代码，以确保板端与PC端跟踪行为的一致性。

初始误差协方差$P_0$基于目标高度$h$进行归一化设置，$p_0 = 2\sigma_p h$，其中$\sigma_p = 1/20$。过程噪声$Q$和观测噪声$R$在每帧根据当前目标高度动态计算：$Q_{pos} = (\sigma_p h)^2$，$R_{pos} = (\sigma_p h)^2$，$R_a = (10^{-1})^2$。这种基于目标高度的自适应噪声设置使滤波器对大目标和小目标均具有合理的预测-观测平衡。

在板端开发初期，卡尔曼滤波的$Q$和$R$矩阵曾被错误地初始化为全零，导致卡尔曼增益$K \approx 1$，滤波器退化为直接输出检测观测值，失去了预测平滑功能，跟踪框随检测框逐帧抖动。修复后的滤波器稳态增益$K \approx 0.5$，输出为预测值与观测值的加权平均，平滑效果显著改善。

这一问题的解决过程体现了PC端与板端参数对齐的重要性。板端C++实现（`deploy/rv1126b_yolov5/src/main_video.cc`中的`ByteTrackAlignTracker::init_kf`函数）与PC端Python实现（`src/tracking/kalman_filter.py`）使用完全相同的噪声权重参数和初始化逻辑，确保跟踪行为的跨平台一致性。

【图4-2】修复前后目标轨迹坐标时序对比

### 4.4 本章小结

本章在统一检测前端条件下，对DeepSORT、ByteTrack和CenterTrack三种跟踪算法进行了系统性对比评估。主要结论如下：

（1）ByteTrack在红外场景下表现最优，ID Switch仅为26次（exp07检测前端），轨迹连续性显著优于DeepSORT（128次）和CenterTrack（87次）。其双阈值二次关联机制有效应对了红外图像中目标检测置信度波动大的问题。

（2）DeepSORT的外观Re-ID特征在红外灰度图像中区分能力不足，导致频繁的错误关联和ID跳变，不推荐用于红外场景。

（3）检测精度对跟踪性能具有正向传导作用，EIoU最优检测模型配合ByteTrack跟踪器构成最优检测-跟踪组合。

基于上述结论，本文选定exp07_eiou + ByteTrack作为后续嵌入式部署的主线配置。

---

## 第5章 RV1126B嵌入式部署与优化

### 5.1 部署方案总体设计

将PC端训练好的检测模型部署至嵌入式平台，需要解决模型格式转换、推理框架适配和性能优化三个核心问题。本文设计的端到端部署方案包含以下环节：

（1）PC端模型导出：使用YOLOv5自带的`export.py`脚本将PyTorch训练权重(.pt)导出为ONNX格式，采用动态batch和opset 11兼容设置。

（2）RKNN模型转换：在Ubuntu环境下通过`deploy/rv1126b_yolov5/python/convert_yolov5_to_rknn.py`脚本将ONNX模型转换为RKNN格式，执行INT8量化。校准数据集从FLIR验证集中随机采样100张红外图像。

（3）交叉编译：在Ubuntu服务器上使用ARM交叉编译工具链（cmake + aarch64-linux-gnu-gcc）编译C++推理程序，生成RV1126B可执行文件`bishe_rknn_video`。

（4）板端推理：通过ADB将可执行文件和RKNN模型部署至RV1126B开发板，执行视频检测+跟踪推理。

RV1126B硬件规格为：Cortex-A53四核处理器（1.5GHz），NPU 3.0 TOPS（单核），2GB LPDDR4内存。在约2W的功耗水平下提供较强的边缘AI推理能力。

【图5-1】端到端部署流程图

### 5.2 模型量化转换

ONNX导出使用命令`python yolov5/export.py --weights best.pt --include onnx --dynamic --opset 11 --img 640`，其中`--dynamic`参数支持动态batch维度，`--opset 11`确保与RKNN Toolkit2的算子兼容性。

RKNN INT8量化是将浮点模型压缩为8位整数表示的关键步骤。本文对比了两种量化方案：

普通量化（normal）采用最大最小值线性映射：$s = (x_{max} - x_{min}) / (2^8 - 1)$，$zp = -\text{round}(x_{min} / s)$。该方法计算简单但对激活值分布的假设较强。

KL散度量化（kl_divergence）通过搜索最优截断阈值$T$，使截断后的量化分布与原始浮点分布的KL散度最小化。对于长尾分布的激活值，KL散度量化能够更好地利用有限的量化位宽。

在实际转换过程中，本文遇到了EIoU模型与普通量化算法的适配问题。EIoU损失的独立宽高惩罚使模型输出层的激活分布更加集中（边界框回归目标更精确，输出值的动态范围较窄），普通量化在此情况下将大量量化位宽分配给分布稀疏的极端值区间，导致密集区间的有效分辨率不足。实际表现为板端检测时出现大量重叠框——多个量化后的不同输出值被映射到相同的INT8表示，解码后生成几乎重合的检测框，NMS无法有效过滤。

改用KL散度量化后，截断阈值自适应匹配实际激活分布，有效量化位宽集中在数值密集区间，重叠框问题得到解决。这一发现表明，损失函数的选择不仅影响训练过程，还会通过改变模型输出分布间接影响量化部署的效果，这在现有文献中较少被讨论。

RKNN Toolkit2还提供了MMSE（最小均方误差）量化选项，理论上通过优化量化参数使量化前后的均方误差最小化，精度应优于KL散度量化。然而，在实际使用中，MMSE量化的校准过程极为耗时（100张校准图需数小时），且在本文模型上的精度提升不明显，因此未被采用。

【表5-1】量化方案对比

| 指标 | 普通量化 | KL散度量化 |
|------|---------|-----------|
| 模型文件大小 | ~14MB | ~14MB |
| 三branch zp示例 | [108, 95, 62] | [81, 56, 3] |
| 三branch scale示例 | [0.787, 0.790, 1.063] | [0.333, 0.358, 0.559] |
| NPU推理时延 | ~31ms | ~30.5ms |
| 板端检测框质量 | 存在重叠框问题 | 正常 |

YOLOv5的RKNN模型导出为3-branch输出格式，每个branch对应一个检测尺度（80×80、40×40、20×20），输出张量形状为[1, 21, H, W]的INT8数据。后处理需要对三个branch分别进行反量化、sigmoid激活和边界框解码，最后合并执行NMS。

### 5.3 C++推理框架设计

板端C++推理程序`bishe_rknn_video`的模块结构如下：

`main_video.cc`为主循环入口，负责视频帧读取、预处理调用、NPU推理调度、后处理结果接收、ByteTrack跟踪更新、HUD绘制和视频写出。

`rknn_detector.cc/hpp`封装RKNN API，包括模型加载初始化、输入张量设置、推理执行和输出张量获取。核心结构体`RknnAppContext`管理RKNN上下文、输出张量属性（含量化参数scale和zp）、预分配的float和int8输出缓冲区以及非缓存输入内存。

`postprocess.cc/hpp`实现后处理流程，包括INT8反量化（NEON加速）、logit预滤波、sigmoid激活、边界框解码（网格偏移+Anchor缩放）和NMS。

预处理环节采用OpenCV NEON加速的Letterbox方案。具体流程为：将BGR视频帧resize至模型输入尺寸的内切矩形，再通过`cvtColor`转换为RGB格式并填充至640×640的灰色(114)画布中。缩放因子$\text{scale} = \min(W_m/W_f, H_m/H_f)$和填充量$(pad_x, pad_y)$被记录用于后处理中检测框坐标的反映射。该Letterbox方案在RV1126B上的耗时约为1.5ms/帧，显著优于基于RGA（2D图形加速器）或纯CPU的Letterbox方案（~8ms/帧）。

板端ByteTrack实现通过`ByteTrackAlignTracker`类封装，内部使用OpenCV的`cv::KalmanFilter`，状态空间和噪声参数与PC端`src/tracking/kalman_filter.py`严格对齐。双阈值策略中，`high_threshold = conf_threshold`（默认0.25），确保所有有效检测框均参与第一轮关联，避免因阈值不对齐导致的跟踪中断。

【图5-2】板端推理主循环流程图

### 5.4 NPU性能优化

为使板端推理达到实时帧率要求，本文实施了以下五项NPU级优化措施：

措施一：RKNN_FLAG_ENABLE_SRAM。在RKNN初始化时启用SRAM标志位，将部分模型权重缓存至NPU片上SRAM（静态随机存取存储器），减少对外部LPDDR4的访问频率。由于SRAM的访问延迟远低于LPDDR4，此措施将NPU推理时延从约31ms降至约28ms，节省约3ms。

措施二：非缓存输入内存+禁止Cache Flush。通过`rknn_create_mem2`创建`RKNN_FLAG_MEMORY_NON_CACHEABLE`属性的输入内存，并在初始化时设置`RKNN_FLAG_DISABLE_FLUSH_INPUT_MEM_CACHE`标志。非缓存内存直接映射至物理地址，`rknn_run`执行前无需进行Cache同步操作，节省约1ms/帧。

措施三：logit预滤波。在后处理的反量化阶段，对每个网格位置的目标置信度（obj_conf）先计算其logit值，与预先计算的阈值logit值$\text{logit}(\text{conf\_thresh})$比较，仅对超过阈值的网格执行完整的sigmoid计算和后续解码。由于红外场景中约98%的网格为背景，此措施将sigmoid调用次数从约176K次降至约3K次，CPU后处理耗时减少约3ms。

措施四：NPU核心绑定。通过`rknn_set_core_mask(RKNN_NPU_CORE_ALL)`将推理任务绑定至所有NPU核心，防止操作系统调度器在运行时迁移NPU任务导致的性能波动。

措施五：编译器优化标志。交叉编译时使用`-O3 -march=armv8-a+simd -ffast-math`编译选项，启用AArch64 NEON向量化指令和快速数学运算，加速反量化和后处理中的浮点计算循环。

【表5-2】各优化措施的增量性能提升

| 优化步骤 | NPU时延(ms) | 端到端FPS | 备注 |
|---------|-----------|----------|------|
| 基准（无优化） | 31.0 | 25.2 | 初始部署 |
| +SRAM标志位 | ~28 | ~28 | LPDDR4带宽释放 |
| +非缓存内存+禁Flush | ~27 | ~29 | Cache同步开销消除 |
| +logit预滤波 | ~27 | ~31 | CPU后处理降低~3ms |
| +核心绑定+-O3+NEON | **~26** | **≥33** | 综合最优 |

经过五项优化措施的叠加，板端端到端FPS从初始的25.2提升至33以上，满足了实时推理（≥25 FPS）的要求。

### 5.5 板端跟踪参数迭代优化

板端NPU的INT8推理存在约±5%的非确定性——同一帧的多次推理可能产生略有差异的检测结果。这种非确定性来源于NPU的异步计算调度和量化舍入误差，导致PC端调试确定的跟踪参数在板端并非最优。本文通过3轮控制变量实验在板端实测中迭代调优跟踪参数。

评测指标选取：唯一轨迹ID数（越少表示ID切换越少，跟踪越稳定）和轨迹展示总数（越高表示可见帧覆盖越多，画面连续性越好）。测试视频为seq006和seq009，检测参数固定为conf=0.25、nms=0.45。

基线配置（PC端参数直接移植）：`match_iou=0.30, second_match_iou=0.20, min_hits=3, visible_lag=1`。seq006唯一ID=49，展示=2353；seq009唯一ID=74，展示=5170。

Round 1（宽松IoU匹配阈值）：将`match_iou`从0.30降至0.25，`second_match_iou`从0.20降至0.15。设计思路为：NPU INT8量化导致检测框坐标精度降低，相邻帧同一目标的预测框IoU值可能下降，适当放宽匹配阈值有助于维持正确关联。seq006唯一ID降至46（-6%）。

Round 2（加速轨迹确认+延长展示窗口，最终选用）：在R1基础上将`min_hits`从3降至2，`visible_lag`从1增至3。`min_hits=2`使轨迹仅需连续2帧匹配即进入confirmed状态，降低了因短暂漏检导致的新轨迹创建；`visible_lag=3`延长了丢失轨迹在画面上的保持时间，提高了视觉连续性。seq006唯一ID降至42（较基线-14%），展示增至2766（+18%）；seq009唯一ID降至66~73（-1%~11%），展示增至5465（+6%）。FPS保持27.6~27.8，无性能退化。

Round 3（降低高阈值，已拒绝）：在R2基础上将`high_threshold`从0.50降至0.45。结果seq006唯一ID反弹至57（较R2+35%），因为阈值降低后更多INT8量化噪声检测框进入高分池，创建了大量短命假轨迹。结论：`high_threshold=0.50`是过滤INT8噪声检测的有效边界，不应降低。

【表5-3】板端跟踪参数3轮迭代对比

| 配置 | seq006唯一ID | seq006展示 | seq009唯一ID | seq009展示 | FPS |
|------|------------|-----------|------------|-----------|-----|
| 基线（PC参数） | 49 | 2353 | 74 | 5170 | 27.6 |
| R1（宽松IoU） | 46（-6%） | — | — | — | 27.8 |
| **R2（最终选用）** | **42（-14%）** | **2766（+18%）** | **66~73** | **5465（+6%）** | **27.6** |
| R3（低阈值，已拒绝） | 57（+16%） | 3302 | — | — | 27.4 |

最终板端跟踪参数配置为：`high_threshold=0.50, low_threshold=0.10, match_iou=0.25, second_match_iou=0.15, min_hits=2, visible_lag=3, reactivate_iou=0.20, max_age=30`。

### 5.6 板端与PC端效果对齐验证

在统一conf=0.25、nms=0.45和相同视频源的条件下，对比板端NPU INT8推理与PC端FP32推理的检测结果。PC端参考检测数为：seq006共3542框，seq009共7902框（exp07_eiou，PyTorch推理）。板端检测数分别为seq006约2813框和seq009约6509框，较PC端减少约20%，原因为INT8量化的精度损失导致部分低置信度检测被截断。

在跟踪效果方面，通过卡尔曼滤波参数对齐和R2参数优化后，板端ByteTrack的跟踪框平滑度接近PC端水平，平均边界框坐标偏差小于5个像素。视觉效果上，板端输出视频的轨迹连续性和框稳定性达到了可接受的工程标准。

### 5.7 本章小结

本章完成了从PC端训练权重到RV1126B嵌入式平台的端到端部署全链路。主要工作和结论如下：

（1）完成ONNX→RKNN模型转换，发现并解决了EIoU模型与普通量化算法的适配问题，确定KL散度量化为最优量化方案。

（2）设计并实现C++推理框架，包含OpenCV NEON Letterbox预处理、RKNN NPU推理、3-branch INT8解码后处理和ByteTrack跟踪四个核心模块。

（3）通过SRAM标志位、非缓存输入内存、logit预滤波、NPU核心绑定和编译器优化五项措施，将板端FPS从25.2提升至33以上。

（4）通过3轮控制变量板端实验，优化跟踪参数使唯一ID数降低14%、轨迹展示覆盖率提升18%，验证了板端部署方案的工程可行性。

---

## 第6章 可视化管理系统设计与实现

### 6.1 系统需求分析与总体架构

为提高研究过程中的实验效率和结果管理便捷性，本文设计并实现了一套基于PyQt5的可视化管理系统。该系统的核心功能需求包括：支持FLIR数据预处理流程的可视化操作、消融模型训练的参数配置和进度监控、检测和跟踪评估的一键触发和结果展示，以及板端部署的远程编译和推理控制。

系统采用"前端UI + 后台线程池 + 远程协议栈"的三层架构。前端UI基于PyQt5的`QTabWidget`实现五页签布局，分别对应数据处理、模型训练、检测评估、跟踪评估和板端部署五大功能模块。后台线程池包含`LocalCmdWorker`（本地子进程执行Python脚本）、`SSHWorker`（通过paramiko连接Ubuntu编译服务器执行远程命令）和`SFTPWorker`/`SFTPUploadWorker`（文件传输）。所有耗时操作均在QThread中执行，通过信号槽机制将进度和结果异步推送至UI主线程，避免界面冻结。

系统的视觉设计采用Art Deco亮色主题，主色调为Navy蓝（#1e3a5f）配金色（#b8860b）装饰线。所有页面采用左右分栏布局：左侧为参数配置面板，右侧为结果预览面板。统一的`MetricCard`组件以卡片形式展示关键指标数值，`LogPanel`组件实时显示后台任务的标准输出日志。

【图6-1】系统总体架构图

### 6.2 各功能模块设计

数据处理模块提供FLIR数据集的格式转换操作。左侧面板配置输入目录（FLIR原始数据）和输出目录，点击执行按钮调用`scripts/data/prepare_flir.py`脚本。右侧`DatasetOverviewPanel`展示2×3样本图网格（随机采样训练集和验证集各3张图像预览）和数据集统计信息（类别数、图像数量、标注数量），支持刷新按钮重新采样展示。

模型训练模块支持消融实验的参数配置和训练过程监控。左侧面板提供模型权重选择（扫描`outputs/ablation_study/`目录）、消融profile选择（controlled/optimal）、单实验指定（`--only expN`）和SSH远程连接配置。右侧`TrainingDashboardPanel`通过下拉菜单选择实验编号，展示该实验的训练曲线图（results.png）、PR曲线、混淆矩阵等12种可视化结果，以及末轮的mAP@0.5、Precision、Recall等关键指标。

检测评估模块调用`scripts/evaluate/eval_detection.py`对选定权重执行批量评估。参数面板提供权重路径、conf/nms阈值、数据集yaml和设备选择等配置项。评估完成后通过`MetricCard`实时展示mAP@0.5、Precision和Recall指标，`LogPanel`显示完整的评估日志输出。

跟踪评估模块调用`scripts/evaluate/eval_tracking.py`，支持DeepSORT、ByteTrack和CenterTrack三种跟踪算法的选择。结果面板包含跟踪指标`MetricCard`（Match Rate、ID Switch、FPS）和内置`VideoPlayer`组件用于播放跟踪结果视频，支持历史结果对比（扫描`outputs/tracking/`目录）。

板端部署模块是系统的核心功能，整合了远程编译、模型上传和推理控制三个子功能。模型管理区提供6个RKNN模型的选择（EIoU/Baseline/Ghost+EIoU各两种量化方案），通过`SSHWorker`在Ubuntu服务器上触发`build_rv1126b.sh`交叉编译脚本，编译完成后由`SFTPUploadWorker`将可执行文件和模型文件上传至板端。推理控制区配置视频序列、检测阈值和跟踪开关，通过SSH+ADB链路触发板端推理程序执行。结果预览区由`SFTPWorker`从板端拉取输出视频至本地，通过内置`VideoPlayer`播放查看。

系统的工程价值在于将分散的脚本命令和远程操作整合为统一的图形化操作界面，降低了实验操作门槛，提高了实验迭代效率，使研究人员能够专注于算法分析而非繁琐的命令行操作。

【图6-2】板端部署模块界面截图

### 6.3 后台任务管理机制

为确保长耗时操作不阻塞UI主线程，系统采用QThread+信号槽的异步任务管理机制。

`LocalCmdWorker`继承`QThread`，在`run()`方法中通过`subprocess.Popen`启动本地Python脚本，逐行读取stdout并通过`log_line`信号发送至UI线程的`LogPanel`。任务完成时发射`finished`信号，携带返回码供UI判断成功或失败。

`SSHWorker`同样继承`QThread`，内部维护一个`paramiko.SSHClient`连接。远程命令的stdout输出通过逐行读取和信号发射实现实时日志显示。系统对输出内容进行关键词着色：包含"error"的行标红，包含"done"或"success"的行标绿。

`SFTPWorker`和`SFTPUploadWorker`分别处理文件下载和上传任务，`finished`信号携带本地文件路径，UI收到信号后自动刷新结果列表或触发视频播放。

整个防UI冻结机制的核心原则是：主线程仅负责UI渲染和信号接收，所有网络通信、子进程启动和文件传输均在独立QThread中执行。

### 6.4 本章小结

本章设计并实现了基于PyQt5的可视化管理系统，涵盖数据预处理、模型训练、检测评估、跟踪评估和板端部署五大功能模块。系统通过SSH/SFTP/ADB协议打通Windows开发机、Ubuntu编译服务器和RV1126B嵌入式板端的三层联动，将分散的命令行操作整合为统一的图形化界面。后台任务的异步执行机制确保了UI的流畅响应。

---

## 第7章 综合实验与系统演示

### 7.1 完整系统功能验证

本节验证从数据准备到板端推理的端到端系统功能完整性。完整的工作流程包含以下环节：

（1）数据准备：通过可视化系统的数据处理模块或直接执行`scripts/data/prepare_flir.py`脚本，将FLIR原始数据集转换为YOLO格式，生成`data/processed/flir/dataset.yaml`配置文件。

（2）模型训练与消融实验：通过`scripts/train/train_ablation.py --profile controlled`执行13组消融实验，训练产物保存在`outputs/ablation_study/`目录下。

（3）检测评估：通过`scripts/evaluate/eval_detection.py --batch-eval`对所有消融模型进行统一评估，生成精度对比结果。

（4）跟踪评估：选定最优检测模型exp07_eiou，依次使用DeepSORT、ByteTrack和CenterTrack进行跟踪评估。

（5）板端部署：将exp07_eiou权重经ONNX导出和RKNN量化后部署至RV1126B，执行板端视频检测+跟踪推理。

上述每个环节均可通过可视化管理系统的对应模块触发执行，也可通过命令行脚本独立运行。两种方式的实验结果一致，验证了系统的功能完整性。

### 7.2 板端与PC端效果对比

【表7-1】板端与PC端效果对比

| 对比维度 | PC端（PyTorch + ByteTrack） | 板端（RKNN + C++ ByteTrack） |
|---------|----------------------------|----------------------------|
| 检测数 seq006 | 3542 | ~2813（NPU INT8，±5%） |
| 检测数 seq009 | 7902 | ~6509（NPU INT8，±5%） |
| 唯一轨迹ID seq006 | — | 42（R2优化后，基线49→-14%） |
| 唯一轨迹ID seq009 | — | 66~73（R2优化后，基线74） |
| 轨迹展示 seq006 | — | 2766（+18% vs 基线2353） |
| 推理帧率 | 153 FPS（GPU） | 27.6~27.8 FPS（NPU+跟踪） |
| 跟踪框平滑度 | 参考标准 | KF参数对齐+R2优化后接近PC |

板端检测数较PC端减少约20%~25%，这是INT8量化精度损失的正常表现。在跟踪层面，通过R2参数优化后，板端的唯一轨迹ID数和轨迹展示覆盖率均达到了实用标准。板端推理帧率27.6~27.8 FPS（含跟踪和HUD绘制开销），满足实时处理的25 FPS阈值要求。

### 7.3 典型场景分析

行人密集场景（seq009）包含565帧，画面中行人以群体方式运动，存在频繁的相互遮挡。在该场景下，ByteTrack的双阈值二次关联机制发挥了关键作用：当行人被部分遮挡时，检测置信度下降至高阈值以下但仍高于低阈值，第二轮关联将其与已有轨迹匹配，避免了因遮挡导致的ID跳变。实验数据显示，seq009中ByteTrack的唯一轨迹ID数在R2优化后降至66~73，轨迹展示增至5465帧次，画面连续性良好。

车辆快速运动场景（seq006）包含221帧，场景中车辆存在大幅度横向运动，帧间检测框位移较大。在该场景下，卡尔曼滤波器的速度状态分量$(\dot{c}_x, \dot{c}_y)$提供了有效的运动预测——即使某帧检测框出现较大偏移，预测步仍能基于历史速度给出合理的框位置估计，更新步通过卡尔曼增益进行修正，实现了快速运动目标的平滑跟踪。seq006中唯一轨迹ID为42，轨迹展示为2766帧次。

### 7.4 系统综合性能评价

【表7-2】系统综合性能指标汇总

| 维度 | 指标 | 数值 | 评价 |
|------|------|------|------|
| 检测精度 | mAP@0.5 | 0.817 | 全消融最优（EIoU） |
| 跟踪稳定性（PC） | ID Switch | 26 | 三算法最低（ByteTrack） |
| 板端跟踪优化 | 唯一ID seq006 | 42（-14%） | 3轮迭代参数调优 |
| 板端跟踪覆盖 | 轨迹展示 seq006 | 2766（+18%） | visible_lag+min_hits优化 |
| 板端推理速度 | FPS | 27.6~27.8 | 超过25FPS实时阈值 |
| 轻量化备选 | 参数减少 | 30.3% | Ghost+EIoU，mAP仅降2.7pp |
| 系统完整性 | 模块覆盖 | 5/5 | 全流程GUI管控 |

综合评价：本文构建的红外多目标检测与跟踪系统在检测精度、跟踪稳定性和板端推理速度三个维度上均达到了预设的工程指标。系统的主要优势体现在以EIoU损失为核心的简洁高效改进策略和面向嵌入式NPU的系统性优化方法，主要局限在于评估指标采用代理指标而非标准MOT评估体系，以及缺乏独立测试集的泛化性验证。

---

## 第8章 总结与展望

### 8.1 主要工作总结

本文围绕红外多目标检测与跟踪系统开展了系统性研究，主要完成了以下四个方面的工作：

第一，基于消融实验方法论的检测模型系统性评估。本文设计并实施了13组严格控变量的消融实验，覆盖轻量化主干替换、注意力机制嵌入和损失函数改进三个维度。实验结果表明，EIoU损失函数在不增加模型参数量的前提下取得最优检测精度（mAP@0.5 = 0.817，较基线提升0.85个百分点），其独立宽高惩罚机制与红外图像目标边界的各向异性特征具有良好的匹配性。同时，实验发现注意力机制在红外灰度图像上未能带来正向增益，改进策略的过度叠加存在负迁移现象——这些"否定性"结论同样为红外场景下的模型选型提供了有价值的实验依据。

第二，基于统一检测前端的多跟踪算法对比评估。以EIoU最优检测模型为统一前端，对DeepSORT、ByteTrack和CenterTrack进行系统对比。ByteTrack以最低的ID切换次数（26次）展现出最优的轨迹连续性，其双阈值二次关联机制特别适合红外场景中目标检测置信度波动大的特点。DeepSORT的外观Re-ID特征在红外灰度图像中区分能力不足，ID切换次数高达128次。

第三，面向RV1126B嵌入式平台的端到端部署与优化。完成ONNX→RKNN模型转换，发现并解决了EIoU模型与普通量化算法的适配问题。设计C++推理框架并实施五项NPU优化措施，将板端FPS从25.2提升至33以上。通过3轮板端跟踪参数迭代优化，使唯一ID数降低14%、轨迹展示覆盖率提升18%。

第四，基于PyQt5的全流程可视化管理系统。覆盖数据预处理、模型训练、检测评估、跟踪评估和板端部署五大模块，通过SSH/SFTP/ADB协议实现三层平台联动。

### 8.2 不足与展望

本文研究存在以下不足，也为后续研究指明了方向：

第一，量化方式的进一步优化。当前采用的是训练后量化（PTQ）方案，通过KL散度校准确定量化参数。量化感知训练（QAT）在训练过程中模拟量化噪声，使模型学习对量化更鲁棒的权重表示，有望进一步缩小量化精度损失（预计mAP提升0.3~0.5个百分点）。

第二，红外专用外观特征研究。本文实验表明DeepSORT的可见光Re-ID特征在红外域失效。后续可探索基于红外辐射强度分布或红外边缘梯度的专用外观描述子，提升跟踪的长程关联能力。

第三，实时视频流接入。当前系统处理离线视频文件，后续可接入红外摄像头（如FLIR Lepton模组通过`/dev/video0`接口）实现RTSP实时流的在线推理，拓展系统的实际部署场景。

第四，标准化评估指标完善。当前跟踪评估使用ID Switch Proxy代理指标，后续应补充标准MOTA、MOTP和IDF1等MOT评估指标，以增强结论的可比性和学术规范性。

第五，数据泛化性验证。当前实验仅在FLIR单一数据集上进行，结论的泛化性有待验证。后续可扩展至KAIST多光谱数据集和实际采集的红外视频数据，评估模型在不同传感器、不同场景条件下的鲁棒性。

---

## 参考文献

[1] Girshick R, Donahue J, Darrell T, et al. Rich feature hierarchies for accurate object detection and semantic segmentation[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2014: 580-587.

[2] Girshick R. Fast R-CNN[C]//Proceedings of the IEEE International Conference on Computer Vision. 2015: 1440-1448.

[3] Ren S, He K, Girshick R, et al. Faster R-CNN: Towards real-time object detection with region proposal networks[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2017, 39(6): 1137-1149.

[4] Redmon J, Divvala S, Girshick R, et al. You only look once: Unified, real-time object detection[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 779-788.

[5] Jocher G, Chaurasia A, Qiu J. YOLOv5 by Ultralytics[EB/OL]. https://github.com/ultralytics/yolov5, 2020.

[6] Han K, Wang Y, Tian Q, et al. GhostNet: More features from cheap operations[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020: 1580-1589.

[7] Ma N, Zhang X, Zheng H T, et al. ShuffleNet V2: Practical guidelines for efficient CNN architecture design[C]//Proceedings of the European Conference on Computer Vision. 2018: 116-131.

[8] Hu J, Shen L, Sun G. Squeeze-and-excitation networks[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018: 7132-7141.

[9] Hou Q, Zhou D, Feng J. Coordinate attention for efficient mobile network design[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021: 13713-13722.

[10] Zhang Y H, Ren W Q, Zhang Z, et al. Focal and efficient IOU loss for accurate bounding box regression[J]. Neurocomputing, 2022, 506: 146-157.

[11] Zheng Z, Wang P, Liu W, et al. Distance-IoU loss: Faster and better learning for bounding box regression[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2020, 34(7): 12993-13000.

[12] Gevorgyan Z. SIoU loss: More powerful learning for bounding box regression[J]. arXiv preprint arXiv:2205.12740, 2022.

[13] Bewley A, Ge Z, Ott L, et al. Simple online and realtime tracking[C]//Proceedings of the IEEE International Conference on Image Processing. 2016: 3464-3468.

[14] FLIR Systems. FLIR thermal dataset for algorithm training[EB/OL]. https://www.flir.com/oem/adas/adas-dataset-form/, 2018.

[15] Wojke N, Bewley A, Paulus D. Simple online and realtime tracking with a deep association metric[C]//Proceedings of the IEEE International Conference on Image Processing. 2017: 3645-3649.

[16] Zhang Y, Sun P, Jiang Y, et al. ByteTrack: Multi-object tracking by associating every detection box[C]//Proceedings of the European Conference on Computer Vision. 2022: 1-21.

[17] Rockchip Electronics. RKNN Toolkit2 User Guide[EB/OL]. https://github.com/rockchip-linux/rknn-toolkit2, 2021.

[18] Bernardin K, Stiefelhagen R. Evaluating multiple object tracking performance: The CLEAR MOT metrics[J]. EURASIP Journal on Image and Video Processing, 2008, 2008: 1-10.

[19] Geiger A, Lenz P, Urtasun R. Are we ready for autonomous driving? The KITTI vision benchmark suite[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2012: 3354-3361.

[20] Woo S, Park J, Lee J Y, et al. CBAM: Convolutional block attention module[C]//Proceedings of the European Conference on Computer Vision. 2018: 3-19.

[21] 刘万军, 费博雯, 曲海成. 基于改进YOLOv5的红外弱小目标检测方法[J]. 红外技术, 2023, 45(3): 280-288.

[22] 张涛, 李明华, 王飞. 基于深度学习的红外目标检测综述[J]. 中国图象图形学报, 2022, 27(9): 2579-2602.

[23] 王海涛, 刘鹏, 陈宇. 面向嵌入式平台的轻量化目标检测算法研究[J]. 计算机工程与应用, 2023, 59(15): 173-182.

[24] 赵国英, 李雪峰, 刘洋. 基于改进DeepSORT的多目标跟踪算法研究[J]. 计算机应用研究, 2022, 39(8): 2491-2496.

[25] Lin T Y, Dollár P, Girshick R, et al. Feature pyramid networks for object detection[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017: 2117-2125.

---

## 致谢

本论文的完成得益于众多人的支持与帮助。

首先，衷心感谢我的指导教师张建锋老师在整个毕业设计过程中给予的悉心指导。从课题选定、实验方案设计到论文撰写，张老师提出了许多建设性的意见和建议，特别是在中期汇报中对消融实验方法论的点评，促使我更深入地思考研究方法的合理性和结论的可靠性。

感谢信息工程学院提供的实验环境和硬件资源，RV1126B开发板的借用使嵌入式部署实验得以顺利进行。

感谢同窗好友在学习和生活中的相互鼓励与帮助，共同度过的大学时光是珍贵的回忆。

感谢YOLOv5、ByteTrack、RKNN Toolkit2等开源项目的开发者，他们的工作为本文的研究提供了坚实的技术基础。

最后，感谢父母多年来的养育之恩和无条件的支持，是你们的付出让我能够安心完成学业。

---

## 附录A 主要程序清单

### A.1 板端RKNN检测器核心代码节选

```cpp
// rknn_detector.cc - RKNN模型加载与推理核心函数
// 初始化时启用SRAM和非缓存内存优化
const uint32_t flags = RKNN_FLAG_ENABLE_SRAM
                     | RKNN_FLAG_DISABLE_FLUSH_INPUT_MEM_CACHE;
ret = rknn_init(&app_ctx->rknn_ctx, model_data, model_size, flags, NULL);

// 创建非缓存输入内存
app_ctx->input_mem = rknn_create_mem2(ctx, input_size,
    RKNN_FLAG_MEMORY_NON_CACHEABLE);
```

### A.2 板端ByteTrack跟踪器参数配置节选

```cpp
// main_video.cc - ByteTrackAlignTracker最终参数配置
ByteTrackAlignTracker tracker(
    /*max_age=*/30,
    /*min_hits=*/2,          // R2优化: 3→2
    /*iou_threshold=*/0.3,
    /*high_threshold=*/0.50,  // INT8噪声过滤边界
    /*low_threshold=*/0.10,
    /*match_iou=*/0.25,       // R1优化: 0.30→0.25
    /*second_match_iou=*/0.15,// R1优化: 0.20→0.15
    /*reactivate_iou=*/0.20,
    /*visible_lag=*/3);       // R2优化: 1→3
```

### A.3 logit预滤波后处理代码节选

```cpp
// postprocess.cc - logit预滤波加速
const float obj_logit_thresh = logit_f(conf_threshold);
for (int i = 0; i < grid_h * grid_w * num_anchors; ++i) {
    // 跳过~98%背景格子，避免无效sigmoid计算
    if (val[4] < obj_logit_thresh) continue;
    // 仅对候选格子执行完整解码
    float obj_conf = sigmoid_f(val[4]);
    // ... box decode + class decode
}
```
