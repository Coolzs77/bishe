# 基于轻量化AI算法的红外多目标检测与跟踪方法研究

**作者：张仕卓　　学号：2022013223　　指导教师：张建锋**
**专业：信息工程　　院系：信息工程学院　　年份：2026**

---

## 中文摘要

本文以FLIR多模态数据集中的可见光配准红外灰度图像为研究对象，以YOLOv5s为基线框架，系统开展了13组控制变量消融实验，从轻量化网络结构（Ghost-C3、Shuffle-C3）、注意力机制（SE注意力、坐标注意力）与损失函数改进（SIoU、EIoU）三个维度定量评估各策略在红外灰度场景中的有效性。实验以两阶段设计推进：第一阶段单变量分析揭示EIoU损失函数对边界框回归质量提升最为显著，mAP@0.5达到0.817；第二阶段组合实验验证了EIoU与各结构改进的兼容性。在多目标跟踪对比实验中，ByteTrack以最低ID Switch（26次）表现出最优轨迹连续性。针对RV1126B嵌入式平台，本文完成了ONNX导出、RKNN INT8量化及C++推理工程化部署，平均推理延迟达19.8ms，满足实时性要求，并记录了量化过程中的关键工程问题及解决方案。

**关键词：** 红外图像；目标检测；多目标跟踪；轻量化；消融实验；边缘部署

---

## Abstract

This paper investigates multi-target detection and tracking methods for infrared grayscale images using the FLIR multi-modal dataset. Based on the YOLOv5s framework, a systematic two-stage ablation study comprising 13 controlled-variable experiments is conducted, evaluating lightweight network structures (Ghost-C3, Shuffle-C3), attention mechanisms (SE Attention, CoordAttention), and loss function improvements (SIoU, EIoU) in infrared scenarios. Stage-1 single-variable analysis reveals that EIoU loss achieves the most significant bounding box regression quality improvement, reaching mAP@0.5 of 0.817. Stage-2 combination experiments validate EIoU compatibility with structural improvements. In multi-object tracking comparison, ByteTrack achieves the lowest ID Switch (26), demonstrating superior trajectory continuity. For the RV1126B embedded platform, ONNX export, RKNN INT8 quantization, and C++ inference deployment are completed with an average inference latency of 19.8ms. Key engineering issues during quantization are documented and resolved.

**Keywords:** Infrared image; Object detection; Multi-object tracking; Lightweight network; Ablation study; Edge deployment

---

# 第一章　绪论

## 1.1　研究背景与意义

在智能安防、无人机侦察、自动驾驶辅助感知等领域，视觉目标检测与跟踪正经历从可见光单模态向多模态融合的转型。与可见光图像相比，红外成像技术依托目标自身的材料反射特性或在特定照射条件下的成像方式，能够在低照度、逆光、夜间等复杂光照条件下保持稳定的成像质量，因而在实际工程应用中具有不可替代的补充价值。本文研究的红外图像，特指经过近红外或短波红外成像传感器采集、以灰度形式呈现的图像，与热成像（Thermal Infrared，依赖目标自身热辐射差异）有本质区别：本文所用图像中并不携带目标温度场信息，目标主要依赖形状、纹理和运动特征进行区分，这一特性使得针对热成像图像设计的部分增强策略并不能直接迁移，需要专门评估。

近年来，以YOLO系列为代表的单阶段目标检测算法在检测精度与推理速度的综合表现上取得了显著进步，YOLOv5凭借其模块化设计和完善的训练生态，成为工程界广泛采用的基线框架。然而，将YOLOv5等通用检测模型直接应用于红外灰度图像时，面临以下挑战：其一，红外图像色彩信息缺失，目标与背景之间的对比度差异依赖纹理与形态特征，而主流模型在可见光RGB图像上预训练的特征提取器可能存在域适应不足的问题；其二，边缘计算场景对模型计算量有严格限制，标准YOLOv5s在RV1126B（RK1126 NPU）等资源受限平台上的实时推理需要模型量化和部署工程化支持；其三，面向红外场景的多目标跟踪研究较可见光场景薄弱，主流跟踪算法在红外灰度图像上的适配性尚缺乏系统评估。

本文的研究意义体现在以下三个层面。在算法层面，通过两阶段控制变量消融实验，系统量化了多种轻量化改进策略在红外灰度场景下的有效性边界，为后续针对该场景的模型选型提供有据可查的定量基准；在工程层面，完整实现了从数据集处理、模型训练、检测评估、跟踪系统集成到嵌入式端部署的全链路工程，并记录了量化过程中的关键工程问题；在应用层面，为安防监控、低功耗边缘端红外目标检测跟踪系统的落地提供了可复现的技术路径。

## 1.2　国内外研究现状

**目标检测方向**的发展经历了从传统手工特征到深度卷积神经网络的深刻转变。早期以HOG+SVM、DPM（变形部件模型）为代表的方法依赖人工设计特征，检测精度受限。2014年前后，Girshick等人提出R-CNN系列（R-CNN、Fast R-CNN、Faster R-CNN），将候选区域提取与分类识别统一到CNN框架下，显著提升了检测精度，但两阶段结构的推理速度难以满足实时需求。与此同时，YOLO（You Only Look Once）系列将检测问题建模为单次前向传播的回归问题，以牺牲部分精度为代价换取极大的速度提升。YOLOv3引入多尺度特征融合，YOLOv4在数据增强、激活函数、损失函数等多方面系统性改进，YOLOv5则在工程化和模块化上做了深度优化，成为目前工业界应用最广泛的检测框架之一。针对红外图像的检测研究，Gade等人的综述指出，热成像目标检测与可见光检测在特征层面有本质差异，但面向近红外灰度图像的系统性对比研究相对稀少，大多数工作直接套用可见光方法而缺乏专项适配分析。损失函数方面，IoU系列从最初的Intersection over Union演化至GIoU、DIoU、CIoU，再到SIoU（引入角度因素）和EIoU（分离宽高约束），逐步提升边界框回归的收敛速度与定位精度。注意力机制方面，SE（Squeeze-and-Excitation）网络通过通道重标定建模通道间依赖，CBAM在通道与空间双维度联合注意，CoordAttention则将位置信息嵌入通道注意力，更适合对目标空间位置敏感的任务。轻量化结构方面，GhostNet提出以廉价线性运算生成冗余特征图的Ghost模块，在保持精度的同时大幅降低参数量与FLOPs；ShuffleNet系列通过分组卷积与通道混洗实现高效特征提取，两者均已被集成入YOLOv5的模块化改进框架。

**多目标跟踪方向**的研究遵循"检测+匹配"的主流范式（Tracking-by-Detection）。SORT（Simple Online and Realtime Tracking）以卡尔曼滤波预测轨迹状态，以IoU距离矩阵执行匈牙利算法匹配，以极简设计实现了接近实时的跟踪速度，成为后续大量工作的基线。DeepSORT在SORT基础上引入外观特征（ReID）嵌入，通过Mahalanobis距离与余弦相似度的加权关联有效降低了ID切换频率。ByteTrack提出将低置信度检测框也纳入关联过程（两轮匹配），充分利用被抑制的候选检测结果，在遮挡场景下大幅提升轨迹连续性，在MOT17等标准数据集上取得SOTA性能。CenterTrack则采用以上一帧热图为条件输入的联合检测跟踪框架，在保留时序信息的同时减少对外观特征的依赖。在红外多目标跟踪领域，现有研究相对集中于热成像场景（如行人车辆红外检测），而面向灰度近红外图像的跟踪系统对比研究较为缺乏，尤其是在嵌入式部署约束下的跟踪性能评估几乎是空白。

**嵌入式部署方向**随着神经网络压缩技术的成熟逐渐形成完整工具链。模型量化将浮点参数映射至低比特定点表示（INT8、INT4），可在几乎不损失精度的前提下将模型体积压缩4倍、推理速度提升2～4倍。TensorRT（NVIDIA）、RKNN-Toolkit2（Rockchip）等工具链已实现从PyTorch模型到硬件原生推理格式的完整转换链路。量化校准策略上，KL散度校准（原生于TensorRT，RKNN亦实现）通过寻找使量化分布与浮点分布KL散度最小的截断阈值，在保留分布主体信息的同时截断异常激活值，相较于MinMax（Normal）校准对激活值分布形状更鲁棒。国内瑞芯微RV1126B平台搭载1TOPS算力的NPU，已被广泛应用于安防摄像头、无人机端侧AI等场景，但公开的RKNN部署实践文档较分散，完整部署链路（ONNX导出格式→量化算法选择→C++推理框架→板端性能基准）的系统性记录具有较高工程参考价值。

## 1.3　主要研究内容与创新点

本文的研究主线围绕三个核心问题展开：哪种轻量化改进策略在红外灰度图像场景下最为有效？主流多目标跟踪算法在该场景下的性能差异如何？如何将最优检测模型可靠地部署至RV1126B嵌入式平台？

**研究内容一：基于两阶段控制变量设计的系统性消融实验。** 本文设计了两阶段消融框架：第一阶段（Stage1，实验Exp1～Exp7）在严格控制超参数的前提下，分别单独引入Ghost-C3、Shuffle-C3、SE注意力、坐标注意力、SIoU损失、EIoU损失，定量测量每种改进相对基线的独立贡献；第二阶段（Stage2，实验Exp8～Exp13）在Stage1结论指导下，对表现较好的结构改进与EIoU损失进行组合实验，分析协同效应。全程采用统一训练口径（相同epochs、batch size、图像尺寸、学习率调度），保证横向比较的公平性。

**研究内容二：多目标跟踪算法在红外灰度图像场景下的对比评估。** 以消融实验选出的最优检测模型为前端，接入DeepSORT、ByteTrack、CenterTrack三种主流跟踪算法，在统一的红外视频序列上评估MOTA、MOTP、IDF1、ID Switch等标准MOT指标，给出各算法的适用场景分析。

**研究内容三：面向RV1126B平台的端侧部署工程实践。** 实现从PyTorch权重到RKNN INT8量化模型的完整转换链路，包括ONNX导出格式选择、量化算法对比（Normal/KL散度）、C++推理框架设计与板端性能测试。系统记录了单输出格式ONNX在INT8量化后检测率归零的问题及3-branch输出格式的解决方案，以及EIoU模型与Normal量化不适配问题及KL散度算法的修复方案。

**创新点：**

（1）**两阶段消融实验方法论创新。** 区别于已有工作中随机堆叠改进的实验设计，本文将消融实验显式分为"单变量揭示独立贡献"与"组合实验验证协同效应"两阶段，并配合统一训练口径约束，使红外灰度场景下各改进策略的有效性边界得以定量化，研究结论具有可复现性。

（2）**量化部署工程发现。** 揭示了YOLOv5单输出格式ONNX在RKNN INT8量化后检测率归零的格式兼容性问题，发现了EIoU训练模型与Normal量化校准算法的激活分布不匹配问题，并给出了对应的工程解决方案，为后续类似部署提供了实践参考。

（3）**完整的红外目标检测-跟踪-部署研究链路。** 本文覆盖了从数据集处理、模型训练、检测与跟踪算法评估到嵌入式端部署的全流程，在现有红外灰度图像相关研究中，此类端到端完整链路记录较为少见。

## 1.4　论文组织结构

本文共分七章：第一章绪论说明研究背景、现状与创新点；第二章介绍相关技术基础；第三章详述消融实验设计与检测实验结果分析；第四章介绍多目标跟踪系统设计与实验对比；第五章描述RV1126B嵌入式部署方案与性能测试；第六章简述可视化管理系统；第七章总结全文并展望未来工作方向。

---

# 第二章　相关技术基础

## 2.1　红外图像的成像特性与目标感知差异

理解红外图像的成像机理是开展针对性检测研究的前提。本文使用的红外图像来源于FLIR ADK传感器采集的近红外/短波红外图像，以灰度格式存储，与热成像（Longwave Infrared，LWIR）在成像原理和图像特性上存在显著差异。

热成像（LWIR）成像依赖目标表面辐射的远红外热辐射，目标的温度差异直接映射为图像亮度差异，因此行人在热成像图像中往往呈现高亮的热轮廓。近红外/短波红外图像则依赖目标表面对主动光源（或环境光）的反射，灰度值对应的是反射率而非温度，目标特征主要体现在材质纹理、边缘形状与运动模式。这一区别带来以下检测挑战：

第一，对比度不稳定。在近红外图像中，目标与背景的灰度对比度依赖光照条件，阴影区域目标与背景的对比度可能远低于热成像场景，使得小目标容易与背景融合。

第二，颜色信息缺失。RGB预训练权重中编码的颜色语义特征（如红色车灯、黄色交通信号）在灰度近红外图像中完全失效，模型在迁移至红外场景时需要依赖形状、纹理等更基础的视觉特征。

第三，目标形态相似。行人与车辆在灰度图像中的外观特征差异主要来自整体形状，在目标较小或图像分辨率有限时，形态辨别难度上升。

上述特性决定了针对红外灰度图像的检测方法需要重点关注边界框的定位精度（而非依赖颜色辨别目标类别）和在低对比度背景下的小目标召回率。这也是本文在损失函数改进方向重点评估EIoU的出发点——相比CIoU，EIoU通过分离宽度误差与高度误差的约束，能够更精细地约束边界框与真实框的几何对齐，在目标形状规则性较强（行人竖直形、车辆横向形）的红外场景中具有理论优势。

## 2.2　YOLOv5检测框架

YOLOv5由Ultralytics团队于2020年发布，是YOLO系列中工程化程度最高、生态最完善的版本之一。其整体结构由三部分构成：Backbone（特征提取）、Neck（特征融合）与Head（多尺度检测）。

Backbone采用CSPDarknet53的变体，核心模块为C3（CSP Bottleneck with 3 convolutions），通过跨阶段部分连接（Cross Stage Partial connections）减少计算量同时保持梯度流。在特征提取阶段，网络从输入图像（640×640）依次下采样至P3（80×80）、P4（40×40）、P5（20×20）三个尺度的特征图。

Neck采用PANet（Path Aggregation Network）结构，以自顶向下的FPN（Feature Pyramid Network）通路进行语义信息下传，再辅以自底向上的通路进行定位信息上传，实现多尺度特征融合。

Head采用三个检测头分别在三个尺度上进行边界框回归与类别分类，每个检测头对每个锚框输出包含边界框坐标（x, y, w, h）、置信度及各类别概率的预测向量。

YOLOv5s（small）版本参数量约为7.2M，在640×640输入下GFLOPs约为16.5，是YOLO系列中最适合嵌入式部署的基线配置，本文以此为消融实验出发点。

训练策略方面，YOLOv5采用马赛克数据增强（Mosaic Augmentation）将4张图像拼合为一张训练图像，有效提升小目标的检测能力；混合精度训练（AMP）加速GPU利用效率；余弦退火学习率调度（Cosine LR Annealing）避免局部最优；早停机制（Early Stopping）在验证指标不再提升时终止训练，防止过拟合。

## 2.3　轻量化网络改进技术

**Ghost模块**由华为Noah方舟实验室提出，其核心观察是：常规卷积生成的特征图中存在大量相似的冗余特征，这些特征可以通过廉价的线性变换（如深度可分离卷积）从少量"本征特征图"生成，而无需完整卷积运算。Ghost-C3将YOLOv5 Backbone中的C3模块替换为使用Ghost卷积的版本，在近似保持特征表达能力的前提下，参数量和FLOPs分别降低约30%～40%。

**Shuffle-C3**借鉴ShuffleNetv2的通道混洗（Channel Shuffle）思想，通过分组卷积降低计算量，再通过通道重排促进不同分组间的信息交流。相比Ghost模块，Shuffle-C3在通道间信息流通上更为充分，但参数压缩比略低。

**SE注意力**（Squeeze-and-Excitation）通过全局平均池化压缩空间维度为通道描述符，再经过两个全连接层生成通道权重，以此对特征图的各通道进行重标定，突出对当前任务贡献大的通道，抑制冗余通道。SE模块参数量极小，通常可忽略不计，但能在特征图通道维度引入自适应权重。

**坐标注意力**（CoordAttention）在SE的基础上将空间位置信息纳入通道注意力计算，通过分别在水平与垂直方向执行全局平均池化，生成携带位置信息的通道描述符，使注意力权重同时编码"哪些通道重要"和"目标在哪里"两类信息，对定位精度敏感的任务（如小目标检测、边界框精细回归）理论上更为有利。

## 2.4　损失函数改进

边界框回归损失是目标检测性能的关键环节。标准IoU损失存在三个已知问题：非重叠框梯度为零；IoU相同但几何形状不同的框被等价对待；收敛速度慢。

**GIoU**（Generalized IoU）引入预测框与真实框的最小外接矩形，通过惩罚非重叠区域比例解决无重叠梯度消失问题。**DIoU**（Distance IoU）在GIoU基础上加入预测框中心与真实框中心的归一化距离惩罚，使收敛更快。**CIoU**（Complete IoU）在DIoU基础上进一步加入宽高比一致性约束项。

**SIoU**（Shape IoU）由Gevorgyan等人提出，引入角度损失项显式惩罚预测框与真实框之间的方向偏差，并将形状损失分解为水平和垂直方向的独立约束，适合目标方向规律的场景。

**EIoU**（Efficient IoU）将CIoU中的宽高比约束项拆分为宽度差和高度差两个独立约束，使梯度反传时宽度误差和高度误差各自独立更新，避免CIoU中联合约束导致的梯度干扰。EIoU的损失函数为：

$$\mathcal{L}_{\text{EIoU}} = 1 - \text{IoU} + \frac{\rho^2(b, b^{gt})}{c^2} + \frac{\rho^2(w, w^{gt})}{C_w^2} + \frac{\rho^2(h, h^{gt})}{C_h^2}$$

其中 $\rho^2(b, b^{gt})$ 为中心点欧氏距离的平方，$c^2$ 为最小外接矩形对角线长度的平方，$C_w$、$C_h$ 分别为最小外接矩形的宽度和高度，$\rho^2(w, w^{gt})$、$\rho^2(h, h^{gt})$ 分别为预测宽度与真实宽度之差、预测高度与真实高度之差的平方。

## 2.5　多目标跟踪算法

多目标跟踪（MOT，Multi-Object Tracking）旨在对视频序列中每一帧的目标进行检测，并在帧间建立同一目标的一致身份标识（Track ID）。主流范式为Tracking-by-Detection，即先用检测器获取每帧检测框，再通过数据关联算法将当前帧检测结果与历史轨迹匹配。

**卡尔曼滤波**是轨迹状态预测的标准工具。对每条轨迹维护状态向量（位置、速度等），根据运动模型预测下一帧位置，再用新观测（检测框）更新状态，实现对遮挡或暂时丢失目标的轨迹保持。

**匈牙利算法**（Hungarian Algorithm）在给定代价矩阵（通常为检测框与轨迹预测框的IoU距离）后，求解最优二分匹配，完成检测结果与现有轨迹的配对。

**DeepSORT**在SORT（Kalman+Hungarian+IoU cost）的基础上引入外观特征向量（由ReID网络提取），将关联代价矩阵改为IoU距离与余弦距离的加权组合，在遮挡或快速运动场景下显著降低ID切换频率。其代价矩阵为：

$$d(i, j) = \lambda \cdot d_{\text{Mahalanobis}}(i, j) + (1-\lambda) \cdot d_{\text{cosine}}(i, j)$$

**ByteTrack**的创新在于利用全部检测框（高置信度+低置信度）分两轮匹配：第一轮以高置信度框与现有轨迹匹配；第二轮以低置信度框尝试与未匹配轨迹关联，以此恢复被遮挡目标的轨迹，大幅减少ID Switch。ByteTrack在MOT17数据集上达到80.3 MOTA，显示出在密集遮挡场景下的优越性。

**CenterTrack**将上一帧的热图（heatmap）作为当前帧检测的条件输入，采用联合检测-跟踪框架，以检测器自身的时序特征建立帧间关联，无需额外的ReID模块。CenterTrack对目标运动的建模依赖于特征级别的时序信息，对长时遮挡的处理不如ByteTrack显式的两轮匹配稳健。

## 2.6　边缘部署与模型量化

RKNN-Toolkit2是瑞芯微为其NPU系列（RK3588、RV1126B等）提供的模型转换与量化工具链。其典型工作流为：PyTorch模型→导出ONNX→RKNN-Toolkit2解析ONNX→量化校准→导出.rknn模型→通过librknn推理库在板端运行。

**量化校准**将浮点权重和激活值映射至INT8表示。校准阶段需要提供少量代表性校准图像（通常50～200张），工具链在这些图像上执行前向传播，统计各层激活值分布，并据此确定量化scale和zero-point。

**Normal量化**（MinMax）直接取校准集上各层激活的最大值和最小值确定量化范围，实现简单，但对激活分布的长尾异常值敏感，当分布存在尖锐峰值时可能造成量化精度损失。

**KL散度量化**通过搜索截断阈值T，使得量化分布与原始浮点分布之间的KL散度最小，等效于找到信息损失最小的表示。相比MinMax，KL散度量化对分布的主体信息保留更好，但计算量略高（仍在可接受范围内）。

**MMSE量化**（Minimum Mean Square Error）逐层迭代搜索使量化均方误差最小的量化参数，精度最高，但每层需要多轮前向传播和参数搜索，对较大模型的量化时间可达数小时，工程实践中通常不优先采用。

RV1126B平台搭载Rockchip RK1126 SoC，内置1TOPS算力NPU，支持INT8推理。CPU为四核Cortex-A7，主频最高1.5GHz，集成ISP支持摄像头接入。该平台在安防、无人机载等低功耗场景已大量部署，适合研究目标检测模型的嵌入式实时推理场景。

---

# 第三章　面向红外图像的目标检测方法研究

## 3.1　数据集处理与构建

### 3.1.1　FLIR数据集概述

本文使用FLIR（Forward Looking Infrared）多模态数据集，该数据集由Teledyne FLIR公司发布，包含同步采集的可见光图像与热红外图像，标注类别覆盖行人（person）、自行车（bicycle）、车辆（car）等多类别，适用于自动驾驶与智能安防研究场景。值得注意的是，本文的研究对象并非数据集中的热红外（LWIR）图像，而是经过预处理后的灰度红外图像：将原始热图像转换为灰度格式，以模拟近红外传感器的成像效果，剔除了目标温度场信息的影响，使研究结论更具一般性。

数据集的类别分布存在一定的不平衡性。训练集中行人类别占比最高，车辆次之，自行车数量相对偏少。为提升模型在主干类别（行人与车辆）上的检测性能，同时控制消融实验的变量数量，本文将检测类别固定为行人（person）与车辆（car）两类，在数据处理阶段过滤掉自行车等低频类别。

数据集划分为训练集、验证集和测试集三部分，最终用于训练的训练集图像数约8000张，验证集约1600张，图像分辨率统一缩放至640×640进行训练。

### 3.1.2　数据预处理流程

数据预处理由 `scripts/data/prepare_flir.py` 脚本完成，主要包括以下步骤：首先读取原始FLIR数据集的JSON格式标注文件，将COCO格式的边界框坐标（xmin, ymin, width, height）转换为YOLOv5所需的归一化中心格式（cx/W, cy/H, w/W, h/H）；其次，根据类别过滤规则保留行人与车辆，丢弃其余类别的标注；随后，将图像从BGR或RGB格式转换为灰度单通道，再复制为三通道灰度（以适配YOLOv5 3通道输入要求）；最后，按训练/验证集划分输出文件列表，并生成 `data/processed/flir/dataset.yaml` 配置文件，指定数据路径、类别数量与类别名称。

处理完成后，数据目录结构如下：

```
data/processed/flir/
├── images/
│   ├── train/     # 训练集图像（.jpg）
│   └── val/       # 验证集图像（.jpg）
├── labels/
│   ├── train/     # YOLOv5格式标签（.txt）
│   └── val/       # YOLOv5格式标签（.txt）
└── dataset.yaml   # 数据集配置文件
```

在数据质量检查阶段，发现少量图像存在无标注（空标签）或标注坐标越界的情况，通过边界裁剪和空标注过滤保证了训练数据的有效性。

## 3.2　消融实验设计方法

本文的消融实验设计遵循"两阶段、控变量、全记录"的方法论原则，具体体现在以下几个方面。

**统一训练口径约束。** 所有消融实验共享相同的训练超参数配置，由 `configs/ablation/train_profile_controlled.yaml` 统一管理：训练轮次（epochs）为100，批次大小（batch size）为16，输入图像尺寸为640×640，余弦退火学习率调度（cos_lr=true），早停阈值（patience）为20轮，数据加载线程数为16，图像缓存模式为RAM缓存。上述配置在消融训练期间对所有实验保持固定，不允许单个实验在profile层面覆盖超参数，以确保横向比较的公平性。

**两阶段实验划分。** 13组实验被显式划分为两个阶段。Stage1（实验Exp1～Exp7）采用单变量控制设计，每个实验仅在基线YOLOv5s之上引入一种改进，其余保持不变；Stage2（实验Exp8～Exp13）在Stage1分析结论的指导下，对筛选出的有效改进进行组合，探究多改进策略的协同效应。两阶段的设计逻辑是：先独立量化每种改进的贡献，再验证组合是否产生叠加效果或相互抑制，避免在对单个改进贡献不清楚的情况下盲目叠加。

**实验编号规范。** 每个实验以 `ablation_expXX_标识` 命名，输出权重保存至 `outputs/ablation_study/ablation_expXX_标识/weights/best.pt`，训练曲线、P/R曲线等图表同步保存，为后续评估和横向比较提供完整证据链。

**评估指标体系。** 检测性能评估采用以下核心指标：
- **Precision（精确率）**：检测框中真正为目标的比例，$P = \text{TP}/(\text{TP}+\text{FP})$；
- **Recall（召回率）**：真实目标中被检测到的比例，$R = \text{TP}/(\text{TP}+\text{FN})$；
- **mAP@0.5**：在IoU阈值0.5下的各类别平均精度均值；
- **mAP@0.5:0.95**：在IoU阈值0.5至0.95（步长0.05）下的各类别平均精度均值，是COCO标准评估指标，对边界框定位精度更敏感；
- **推理延迟（Latency）**：单张图像的GPU推理时间（ms），反映模型计算效率。

## 3.3　Stage1单变量实验结果与分析

Stage1共包含7组实验，覆盖结构改进（Ghost-C3、Shuffle-C3、SE注意力、坐标注意力）与损失函数改进（SIoU、EIoU）两个维度。表3-1汇总了所有Stage1实验在验证集上的核心检测指标。

**表3-1　Stage1单变量消融实验结果对比**

| 实验编号 | 改进策略 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | 推理延迟(ms) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Exp1 | 基线YOLOv5s | 0.823 | 0.763 | 0.806 | 0.497 | 5.2 |
| Exp2 | Ghost-C3 | 0.819 | 0.761 | 0.801 | 0.491 | 4.8 |
| Exp3 | Shuffle-C3 | 0.815 | 0.758 | 0.799 | 0.489 | 4.6 |
| Exp4 | SE注意力 | 0.826 | 0.766 | 0.808 | 0.501 | 5.4 |
| Exp5 | 坐标注意力 | 0.828 | 0.768 | 0.810 | 0.503 | 5.5 |
| Exp6 | SIoU损失 | 0.820 | 0.765 | 0.804 | 0.495 | 5.2 |
| Exp7 | EIoU损失 | **0.835** | **0.776** | **0.817** | **0.516** | 5.2 |

Stage1实验呈现出以下规律性结论：

**轻量化结构改进整体表现平淡，甚至略有下滑。** Ghost-C3（Exp2）和Shuffle-C3（Exp3）在带来推理速度小幅提升（延迟分别降至4.8ms和4.6ms）的同时，mAP@0.5分别相对基线下降了0.5个百分点和0.7个百分点。分析原因：轻量化结构通过减少参数量实现加速，但在目标特征相对单一（灰度红外，缺乏颜色信息）的场景下，特征提取的"冗余"更难以承受——Ghost卷积生成的近似特征图在RGB场景下能有效替代部分真实卷积，但在灰度红外场景中，可用的判别性特征本就有限，进一步压缩特征表达反而损害了检测能力。

**注意力机制带来适度改善。** SE注意力（Exp4）和坐标注意力（Exp5）分别使mAP@0.5提升了0.2和0.4个百分点，推理延迟增加约0.2～0.3ms，代价极小。坐标注意力略优于SE注意力，印证了位置信息对红外目标定位的辅助价值。但整体来看，注意力机制的提升幅度有限，单独引入难以从根本上提升模型的定位精度。

**EIoU损失函数表现最为突出。** Exp7（EIoU）在不增加任何计算量的前提下，将mAP@0.5从基线的0.806提升至0.817（+1.1个百分点），mAP@0.5:0.95从0.497提升至0.516（+1.9个百分点）。mAP@0.5:0.95的大幅提升尤为关键，说明EIoU对边界框定位精度的改善是显著的，而非仅仅提升了宽松IoU阈值下的粗略检测率。SIoU（Exp6）的表现略低于基线，推测SIoU中引入的角度损失项在行人和车辆这类形状规则的目标上角度约束意义有限，反而可能在边框收敛的初期阶段引入额外梯度噪声。

Stage1的核心结论：**EIoU损失是本场景下性价比最高的单一改进策略**，应作为Stage2组合实验的核心成分；坐标注意力具备一定的协同潜力；轻量化结构在当前红外灰度场景和模型规模下贡献有限但可进一步探索。

### 3.3.1　EIoU与SIoU对比分析

EIoU与SIoU都是在CIoU基础上的改进，但设计思路不同，在本实验中的表现分化也值得深入分析。

SIoU的角度损失项设计假设前提是：预测框与真实框之间存在系统性的方向偏差，通过显式惩罚角度误差能加速收敛。然而，FLIR数据集中行人目标主要呈竖直形态，车辆呈横向形态，两类目标的宽高比比较规律，水平/竖直方向的定位误差比角度误差更主要，SIoU的角度约束在此场景中的驱动力不足，甚至可能在框未对齐时引入方向判断的额外噪声。

EIoU的独立宽高约束设计更契合本场景的需求：行人检测中宽度误差和高度误差往往独立出现（高度过长或宽度过宽），EIoU允许梯度分别沿宽度和高度方向回传，使边界框在两个维度上独立收敛，相比CIoU的联合宽高比约束更为灵活。

### 3.3.2　轻量化结构与红外场景的适配性讨论

Stage1中轻量化结构的边际下滑需要从两个角度理解：第一，Ghost-C3的设计前提是"冗余特征可用线性变换近似"，这在信息量丰富的RGB图像中成立，但灰度红外图像的信息密度本身偏低，Ghost近似引入的误差与原始信息量之比更高，导致近似代价更大；第二，当前实验仅替换了C3模块，而Backbone早期层（小感受野卷积）的参数量占比有限，轻量化收益不如在Neck等大Feature Map处操作明显。

需要指出的是，Stage1中轻量化结构的劣势并不意味着轻量化在红外场景毫无价值，而是说明在追求检测精度的Stage1实验框架下，轻量化结构难以在提速的同时保持精度。若在部署强约束下需要以精度换速度，Ghost-C3和Shuffle-C3仍是可行的折衷选项，Stage2将进一步探究其与EIoU的组合效果。

## 3.4　Stage2组合实验结果与分析

Stage2在Stage1结论的基础上，选取若干改进策略进行组合，验证协同效应。6组组合实验的结果见表3-2。

**表3-2　Stage2组合实验结果对比**

| 实验编号 | 改进组合 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | 推理延迟(ms) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Exp8 | Ghost-C3 + SE注意力 | 0.820 | 0.762 | 0.802 | 0.493 | 5.0 |
| Exp9 | Ghost-C3 + EIoU | 0.830 | 0.771 | 0.812 | 0.508 | 4.8 |
| Exp10 | SE注意力 + EIoU | 0.836 | 0.778 | 0.818 | 0.518 | 5.4 |
| Exp11 | Shuffle-C3 + 坐标注意力 | 0.818 | 0.760 | 0.802 | 0.492 | 4.9 |
| Exp12 | Shuffle-C3 + 坐标注意力 + SIoU | 0.816 | 0.759 | 0.800 | 0.490 | 4.9 |
| Exp13 | Shuffle-C3 + 坐标注意力 + EIoU | 0.829 | 0.770 | 0.812 | 0.507 | 4.9 |

Stage2实验揭示了以下重要规律：

**EIoU与注意力机制具有正向协同效应。** Exp10（SE注意力+EIoU）的mAP@0.5达到0.818，mAP@0.5:0.95达到0.518，均略优于Exp7（单独EIoU），显示两种改进在性质上互补：注意力机制提升特征通道的判别性，EIoU提升边界框的定位精度，两者协同作用有效。

**EIoU弥补了轻量化结构的精度损失。** Exp9（Ghost-C3+EIoU）的mAP@0.5为0.812，虽低于纯EIoU的0.817，但显著优于纯Ghost-C3的0.801，推理延迟保持在4.8ms（低于基线），说明当部署场景对推理速度有额外要求时，Ghost-C3+EIoU组合是一个较好的速度-精度折衷方案。

**SIoU对组合实验的贡献持续为负。** Exp12（Shuffle-C3+坐标注意力+SIoU）的mAP@0.5（0.800）低于Exp11（0.802），进一步印证了SIoU在当前场景下贡献有限的Stage1结论，不宜纳入最优配置。

**最优单点配置确定为Exp7（EIoU）。** 综合考量检测精度（mAP@0.5=0.817，mAP@0.5:0.95=0.516）、推理延迟（5.2ms）与模型复杂度（参数量与基线相同），Exp7在不增加任何计算开销的前提下实现了最显著的精度提升，具备最高性价比。如需在精度与速度间折衷，Exp9（Ghost-C3+EIoU）为次优选项。

后续跟踪实验与嵌入式部署均以Exp7（EIoU）最优权重为前端检测模型。

## 3.5　检测结果可视化分析

为定性验证最优模型的检测效果，在FLIR验证集的典型场景上对比了基线（Exp1）与最优配置（Exp7）的检测结果。

在行人密集场景中，基线模型存在两类典型误差：（1）小目标行人的边界框偏大，覆盖了部分背景区域；（2）部分行人目标的置信度分数处于阈值边缘，在后处理时被过滤。EIoU模型在同一场景下，边界框与真实目标的对齐程度明显更好，小目标行人的定位精度显著提升，置信度分数整体更高（与量化实验中的激活值分布结论呼应）。

在车辆检测场景中，基线模型对于横向停放或行驶的车辆，边界框的宽度估计偏差较大；EIoU模型的宽度预测更为准确，这与EIoU独立约束宽高方向的机制设计一致。

总体而言，mAP@0.5:0.95的提升（+1.9个百分点）在可视化结果上表现为更精确的边界框定位，而非仅仅增加检测框的数量，这是EIoU改进的核心价值所在。


---

# 第四章　多目标跟踪系统设计与实验分析

## 4.1　检测-跟踪系统总体架构

多目标跟踪系统采用Tracking-by-Detection的两阶段范式，其整体架构可分为三个功能层：目标检测层、状态管理层与数据关联层。

**目标检测层**以消融实验选出的最优检测模型（Exp7，EIoU损失YOLOv5s）作为前端，负责对输入视频帧逐帧执行目标检测，输出当前帧中所有目标的边界框坐标、置信度分数和类别标签。检测层的输出质量直接决定跟踪系统的上限——检测召回率不足会导致目标轨迹频繁中断，检测精确率不足则会引入虚假轨迹，增加跟踪器的匹配负担。

**状态管理层**维护当前所有活跃轨迹的状态信息，包括每条轨迹的唯一ID、当前位置（卡尔曼滤波预测位置）、外观特征（仅DeepSORT使用）、连续未匹配帧计数（用于轨迹终止判断）等。对于每一帧，状态管理层首先通过运动模型预测各轨迹在当前帧的位置，作为数据关联的参考；匹配完成后更新各轨迹状态，并根据匹配结果决定是否新建轨迹或终止超时轨迹。

**数据关联层**负责将当前帧的检测结果与历史轨迹进行匹配，是三种跟踪算法（DeepSORT、ByteTrack、CenterTrack）的核心差异所在。关联策略的设计决定了系统在遮挡、目标交叉等复杂场景下的鲁棒性。

跟踪系统的完整处理流程如下：视频帧输入→检测器推理→NMS后处理得到检测框列表→轨迹状态预测→代价矩阵计算→匈牙利算法匹配→轨迹状态更新（匹配/新建/终止）→输出带ID的跟踪结果→写入结果文件。

## 4.2　跟踪算法实现与配置

本文在统一的跟踪框架下实现了三种跟踪算法，共享检测器前端和评估流程，仅关联策略和超参数不同，保证了实验对比的公平性。跟踪系统由 `src/tracking/` 模块实现，采用 `BaseTracker` 抽象基类定义统一接口，各算法继承基类并实现各自的 `update(detections)` 方法。

**DeepSORT实现。** 在标准SORT（Kalman+IoU关联）基础上，DeepSORT引入了基于ReID网络的外观特征提取。本文使用在MARS（行人重识别数据集）上预训练的轻量级ReID特征提取器，特征维度为128维。关联代价矩阵由Mahalanobis距离（运动约束）与余弦距离（外观相似度）加权组合，其中Mahalanobis距离作为门限过滤（距离超过阈值的配对直接标记为不可匹配），余弦距离作为精细匹配的主要依据。轨迹管理参数：新轨迹需连续出现n_init=3帧才被确认为活跃轨迹，轨迹连续max_age=30帧未匹配则被终止。

**ByteTrack实现。** ByteTrack的核心在于两轮匹配策略。第一轮：将检测框按置信度阈值（0.5）划分为高置信度集合和低置信度集合，将高置信度检测框与当前所有活跃轨迹进行IoU距离匹配（匈牙利算法）；第二轮：将低置信度检测框与第一轮未匹配的轨迹（可能因遮挡导致置信度下降）进行IoU匹配，成功关联的轨迹得以保留而非被错误终止。未匹配高置信度检测框初始化新轨迹，连续三帧出现后确认。ByteTrack不使用外观特征，仅依赖IoU距离，计算效率高，在遮挡恢复场景下表现尤为突出。

**CenterTrack实现。** CenterTrack不同于SORT和ByteTrack，它是一种联合检测-跟踪框架，将上一帧的检测热图和位移图作为当前帧的条件输入。本文在适配阶段对CenterTrack做了以下调整：以EIoU检测模型替换原始CenterTrack的检测头（以保持与其他两种算法的可比性），在运动关联阶段保留CenterTrack的特征级时序信息。CenterTrack的时序建模依赖连续帧间的特征级对齐，对目标运动平滑性的假设较强，在目标快速转向或帧间运动幅度大时容易产生关联错误。

跟踪评估由 `scripts/evaluate/eval_tracking.py` 脚本执行，配置文件为 `configs/tracking_config.yaml`，支持命令行参数覆盖，可指定跟踪算法、权重路径、输出目录等。

## 4.3　评估指标体系

多目标跟踪评估采用MOT领域标准指标，计算工具基于py-motmetrics库实现。

**MOTA**（Multiple Object Tracking Accuracy，多目标跟踪精度）是综合反映跟踪系统整体性能的核心指标，计算公式为：

$$\text{MOTA} = 1 - \frac{\sum_t(\text{FN}_t + \text{FP}_t + \text{IDSW}_t)}{\sum_t \text{GT}_t}$$

其中 $\text{FN}_t$ 为第t帧漏检数，$\text{FP}_t$ 为虚假检测数，$\text{IDSW}_t$ 为ID切换次数，$\text{GT}_t$ 为真实目标数。MOTA值越高（上限为1.0）表示跟踪整体性能越优。

**MOTP**（Multiple Object Tracking Precision，多目标跟踪精度）衡量匹配成功的轨迹与真实目标框的空间对齐精度，即所有成功匹配对的IoU均值，反映轨迹定位准确性。

**IDF1**（ID F1 Score）综合衡量轨迹的身份关联能力，结合轨迹的精确率和召回率，对轨迹在整个视频序列中的连续性敏感，是比MOTA更能反映ID维持能力的指标。

**ID Switch（IDSW）** 统计整个评估序列中同一真实目标的跟踪ID发生变化的次数，是衡量轨迹稳定性最直接的指标，越低越好。

**MT（Mostly Tracked）** 统计被轨迹覆盖超过80%生命周期的真实目标数量占比；**ML（Mostly Lost）** 统计轨迹覆盖少于20%的真实目标数量占比。

## 4.4　跟踪实验结果与分析

跟踪实验在FLIR验证集的视频序列上执行，以最优检测模型（Exp7）为统一前端。考虑到跟踪实验对存储和时间开销较大，评估时关闭视频保存（--no-save-vid）和轨迹文件写入（--no-save-txt），以降低IO负载，并启用半精度推理（--half）加速前端检测。表4-1汇总了三种算法的MOT评估结果。

**表4-1　多目标跟踪算法对比实验结果**

| 跟踪算法 | MOTA↑ | MOTP↑ | IDF1↑ | ID Switch↓ | MT↑ | ML↓ | 推理FPS↑ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| DeepSORT | 0.682 | 0.741 | 0.658 | 54 | 0.61 | 0.18 | 18.3 |
| ByteTrack | **0.714** | **0.756** | **0.703** | **26** | **0.67** | **0.15** | **24.1** |
| CenterTrack | 0.671 | 0.728 | 0.631 | 71 | 0.58 | 0.22 | 16.8 |

实验结果在多个维度上均呈现出清晰的规律性：

**ByteTrack在各项主要指标上均表现最优。** MOTA达到0.714，比DeepSORT高3.2个百分点，比CenterTrack高4.3个百分点；MOTP为0.756，反映出轨迹定位精度最高；IDF1为0.703，说明ByteTrack在整个序列上的身份关联能力最强；ID Switch仅26次，远低于DeepSORT（54次）和CenterTrack（71次），证明ByteTrack的两轮匹配策略在目标遮挡场景下能有效避免轨迹中断引发的ID重分配。推理FPS为24.1帧，满足实时跟踪要求，且无需外观特征提取，计算开销最小。

**DeepSORT性能居中，外观特征的价值受场景限制。** DeepSORT的ID Switch（54次）明显多于ByteTrack，原因在于：在红外灰度图像中，目标的外观特征（ReID特征）区分能力弱于可见光场景——灰度图像中行人的服装颜色信息消失，车辆的颜色标识消失，仅凭形态轮廓进行ReID特征匹配时，相似目标之间的特征距离较小，余弦相似度区分性下降。此外，ReID特征提取在每帧对每个目标均需前向传播，增加了额外的计算开销，使FPS降至18.3。

**CenterTrack表现最弱，时序建模假设与红外灰度场景不匹配。** CenterTrack的ID Switch最多（71次），MOTA最低（0.671），原因分析如下：CenterTrack依赖上一帧热图的特征级时序信息，该机制在像素级目标特征丰富（彩色、高对比度）的场景下能有效捕捉目标运动，但在灰度红外图像中，目标与背景的低对比度使热图特征噪声更大，时序对齐误差增加，导致关联失效率上升。此外，CenterTrack对于快速运动目标（视频帧间位移大）的预测误差较大，容易丢失轨迹后重新赋予新ID。

**综合评估结论：ByteTrack在红外灰度图像多目标跟踪场景中最为适用。** 其不依赖外观特征（规避了灰度图像ReID能力弱的问题）、两轮匹配策略充分利用低置信度检测框（缓解了低对比度场景下置信度下降导致的目标丢失）、计算效率最高（满足嵌入式部署实时性要求），是当前研究场景下的最优选择。

## 4.5　跟踪参数敏感性分析

在确定最优算法（ByteTrack）后，对其核心超参数进行了简单的灵敏度分析，以评估参数设置的稳健性。

高置信度检测框阈值（high_thresh）直接影响参与第一轮匹配的检测框数量。实验中尝试了0.4、0.5和0.6三个阈值：设置为0.4时，更多检测框进入第一轮匹配，MOTA有小幅提升（+0.5%），但FP数量上升；设置为0.6时，部分真实目标因置信度未达阈值被降级为低置信度集合，依赖第二轮匹配找回，ID Switch无明显变化，但MOTA轻微下降；综合考量，0.5的默认设置提供了最均衡的性能。

轨迹最大缺失帧数（max_age）控制轨迹在未匹配多少帧后被终止。设置为30帧（默认）时，系统能对短时遮挡（约1秒）保持轨迹存续，遮挡结束后ID保持不变；设置为15帧时，短时遮挡恢复后ID切换增加约8次；设置为50帧时，僵尸轨迹（实际已离场的目标轨迹）增加，FP上升。最终采用max_age=30作为默认配置。

---

# 第五章　RV1126B嵌入式部署方案

## 5.1　部署需求分析与总体设计

RV1126B嵌入式平台作为本文目标检测系统的最终部署载体，需要在资源严格受限的条件下实现接近实时的推理性能。在开展部署工程化工作之前，首先进行了系统性的需求分析与方案设计。

**硬件资源约束。** RV1126B平台搭载Rockchip RK1126 SoC，其计算资源分配如下：NPU（Neural Processing Unit）提供1TOPS（INT8）的神经网络加速算力，是深度学习推理的主要载体；CPU为四核ARM Cortex-A7，最高主频1.5GHz，用于前处理、后处理及业务逻辑；RAM通常配置为1GB DDR4，需在操作系统、应用程序与模型权重之间合理分配；NPU推理时最大支持INT8量化模型，浮点模型（FP32/FP16）只能在CPU上运行，速度远低于NPU路径。

**实时性目标。** 针对安防监控场景，本文设定20ms（≈50FPS）为目标推理延迟上限。由于嵌入式平台的NPU存在加速比固定（相比GPU弹性更小）的特性，实现20ms目标需要同时满足：模型量化至INT8（利用NPU加速）、合理的推理输入分辨率（640×640为上限）、高效的C++推理框架（避免Python解释器开销）。

**部署总体链路。** 完整部署链路分为PC端准备阶段和板端运行阶段两部分：PC端（Ubuntu 20.04）负责完成ONNX导出、RKNN量化转换与交叉编译；板端（RV1126B，运行Buildroot Linux）负责模型推理与结果输出。两端以.rknn模型文件和编译好的C++可执行文件为交接点。

## 5.2　ONNX导出与格式问题

将PyTorch训练权重（.pt）转换为RKNN支持的格式，必须先经过ONNX（Open Neural Network Exchange）中间格式。YOLOv5提供了内置的ONNX导出脚本（`yolov5/export.py`），但在与RKNN-Toolkit2的对接中，遇到了重要的格式兼容性问题。

**问题一：单输出格式ONNX在INT8量化后检测率归零。** 初次导出时，采用了YOLOv5默认的简化单输出格式（`--include onnx --simplify`），该格式将三个检测头的输出在ONNX图内部进行了合并（Concat+Reshape），输出为单一tensor。将此格式的ONNX送入RKNN-Toolkit2进行INT8量化后，在板端推理时发现所有置信度分数均为0，检测框数量为零，模型完全失效。

**问题分析。** RKNN量化工具链在对ONNX图进行量化时，需要对各算子进行逐层激活值校准。单输出格式中，YOLOv5的三个检测头（对应P3/P4/P5三个尺度）的sigmoid激活与边界框解码被融合进ONNX图的后处理部分，量化工具在处理这些被融合的后处理算子时出现量化参数估计偏差，导致置信度分支的量化精度损失极大（置信度分数的真实值通常在0.1～0.9之间，量化后被截断至0附近）。

**解决方案：使用3-branch输出格式。** 修改ONNX导出参数，指定 `--opset 12` 并不启用simplify，同时手动修改导出逻辑，使三个检测头分别输出（不在ONNX图内合并），由推理端的后处理代码负责合并三个尺度的输出并执行NMS。3-branch格式中，ONNX图以检测头的原始sigmoid输出为终点，后处理完全在推理端的C++代码中实现，RKNN量化工具链只需对特征提取和检测头卷积进行量化，激活值分布清晰，量化后检测效果恢复正常。

具体导出命令为：
```bash
python yolov5/export.py \
    --weights outputs/ablation_study/ablation_exp07_eiou/weights/best.pt \
    --include onnx \
    --opset 12 \
    --img 640 \
    --batch 1
```

## 5.3　RKNN量化转换与量化算法选择

ONNX模型成功导出后，使用RKNN-Toolkit2完成INT8量化转换。量化转换分为配置加载、模型解析、量化校准和RKNN导出四个阶段。

**校准数据集准备。** 从FLIR验证集中均匀采样120张图像作为量化校准集，覆盖行人密集、车辆场景、混合场景等典型分布。校准图像路径写入 `deploy/rv1126b_yolov5/calibration_dataset.txt`，供RKNN-Toolkit2加载。校准集的代表性对量化精度至关重要——若校准集分布偏斜，量化参数可能无法准确反映真实推理场景的激活分布。

**问题二：EIoU模型与Normal量化不适配。** 使用Normal（MinMax）量化算法对Exp7（EIoU）模型进行INT8量化后，板端推理出现大量碎框（密集低置信度检测框），NMS后仍有大量误检，实际可用的检测结果被噪声严重干扰。

**问题分析。** EIoU损失函数通过独立约束宽度误差和高度误差，使模型在边界框精细对齐上训练更充分，其结果是网络预测层各通道的激活值分布相比CIoU基线模型更为集中（均值附近的密度更高，尾部更薄）。Normal量化直接用激活值的全局最大最小值确定量化scale，量化范围被尾部少数激活值拉宽，实际上大量的激活值只占用了可用量化区间的一小部分，导致置信度分数附近的量化分辨率不足，阈值附近的候选框置信度被量化误差干扰，产生大量碎框。

**解决方案：使用KL散度量化算法。** 将量化算法从 `normal` 改为 `kl_divergence`，RKNN-Toolkit2通过搜索最小化KL散度的截断阈值T，使量化分布在信息损失最小的前提下适配激活值的实际分布形态，有效压制了尾部异常值对量化scale的干扰，置信度分支的量化精度得到保证，板端碎框问题消除。

量化转换的核心Python配置如下（由 `deploy/rv1126b_yolov5/python/` 下的转换脚本执行）：

```python
rknn.config(
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],
    target_platform='rv1126',
    quantized_algorithm='kl_divergence',  # 关键：使用KL散度量化
    quantized_dtype='asymmetric_quantized-8',
    optimization_level=3,
    output_optimize=1,
)
rknn.load_onnx(model='best_3branch.onnx', outputs=['output0','output1','output2'])
rknn.build(do_quantization=True, dataset='calibration_dataset.txt')
rknn.export_rknn('./model/yolov5s_eiou_int8_kl.rknn')
```

量化完成后，在PC端模拟器上进行初步推理验证（`rknn.init_runtime(target=None)`），确认检测框输出正常、无碎框后，再将模型文件传输至板端。

## 5.4　C++推理框架设计与实现

为在RV1126B上实现高效的C++推理，设计了一套完整的推理框架，包含图像预处理、NPU推理调用、后处理与结果输出四个模块。

**图像预处理模块。** 输入图像（来自摄像头或文件）经过以下处理步骤后送入NPU：读取图像→转换为灰度格式→扩充为三通道灰度（保持与训练一致）→等比例缩放至640×640（letterbox，保持宽高比，不足部分填充灰色）→归一化至[0,1]（除以255）→数据排列为NCHW格式。预处理代码使用OpenCV实现，编译时链接ARM版OpenCV库。

**NPU推理模块。** 通过librknn_api.so调用RKNN NPU推理接口，主要步骤包括：初始化RKNN上下文（`rknn_init`）、设置输入tensor（`rknn_inputs_set`）、执行推理（`rknn_run`）、获取输出tensor（`rknn_outputs_get`）。由于采用3-branch输出格式，NPU返回三个输出tensor（对应P3、P4、P5三个检测尺度），每个tensor的维度为 $[1, \text{num\_anchors} \times (5 + \text{nc}), H_s, W_s]$，其中nc为类别数（本文为2），$H_s \times W_s$ 对应各尺度特征图分辨率。

**后处理模块。** C++后处理实现包括：sigmoid激活（还原置信度和类别概率）→边界框解码（从偏移量还原绝对坐标，根据letterbox缩放参数还原至原始图像坐标系）→置信度过滤（保留置信度×类别概率超过阈值的候选框）→非极大值抑制（NMS，去除冗余重叠框）→输出最终检测结果（类别、坐标、置信度）。NMS采用基于IoU阈值的贪心策略，默认检测阈值0.25，NMS阈值0.45。

**交叉编译配置。** PC端交叉编译使用RV1126B官方提供的交叉编译工具链（arm-linux-gnueabihf-g++），通过CMakeLists.txt配置目标架构和库依赖（`deploy/rv1126b_yolov5/CMakeLists.txt`）。编译脚本为 `deploy/rv1126b_yolov5/build_rv1126b.sh`，执行后在 `deploy/rv1126b_yolov5/build/` 目录生成可直接在RV1126B上运行的可执行文件。

## 5.5　板端性能测试与分析

将编译好的可执行文件与量化后的.rknn模型通过ADB或SCP传输至RV1126B，在板端执行推理性能测试。测试方案分为延迟测试和精度对比测试两部分。

**延迟测试方法。** 在板端对100张FLIR验证集图像进行连续推理，记录每张图像的完整推理延迟（包含预处理+NPU推理+后处理），计算均值（mean）、中位数（p50）、90分位数（p90）和95分位数（p95）。表5-1汇总了性能测试结果。

**表5-1　RV1126B板端推理性能统计**

| 阶段 | 均值(ms) | p50(ms) | p90(ms) | p95(ms) |
|:---:|:---:|:---:|:---:|:---:|
| 预处理（Letterbox+归一化） | 3.2 | 3.1 | 3.8 | 4.1 |
| NPU推理（INT8） | 12.6 | 12.4 | 13.2 | 13.8 |
| 后处理（解码+NMS） | 4.0 | 3.9 | 4.5 | 4.9 |
| **全流程合计** | **19.8** | **19.4** | **21.5** | **22.8** |

全流程均值延迟为19.8ms，满足设定的20ms目标。NPU推理阶段占全流程延迟的约63.6%，是主要耗时环节；后处理（C++实现的sigmoid+解码+NMS）耗时约4.0ms，在CPU上执行，若后续需要进一步优化，可考虑将后处理的sigmoid和解码部分也纳入NPU计算图。

**精度对比测试。** 将板端INT8量化推理结果（KL散度量化）与PC端FP32推理结果进行比对，评估量化精度损失。表5-2给出了量化前后的检测指标对比。

**表5-2　量化前后检测精度对比（验证集）**

| 推理方式 | 量化算法 | mAP@0.5 | mAP@0.5:0.95 | 平均延迟(ms) | 硬件 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| FP32推理 | 无量化 | 0.817 | 0.516 | 5.2 | GPU(PC) |
| INT8量化 | Normal | 检测失效 | - | - | RV1126B NPU |
| INT8量化 | KL散度 | 0.803 | 0.501 | 19.8 | RV1126B NPU |

KL散度INT8量化相对FP32基线的精度损失为mAP@0.5降低1.4个百分点，mAP@0.5:0.95降低1.5个百分点，处于可接受范围（通常INT8量化精度损失在2个百分点以内为工程可接受）。Normal量化在EIoU模型上完全失效，进一步印证了量化算法选择对特定损失函数训练模型的重要性。

**精度-速度权衡结论。** 与PC端GPU推理相比，RV1126B NPU在功耗约3W（对比GPU数十至数百W）的约束下，实现了接近实时（19.8ms均值延迟）的检测推理，精度损失约1.4个百分点。这一权衡对安防监控、无人机载端侧AI等低功耗应用场景是合理且可接受的。

## 5.6　部署关键工程问题总结

本章的部署工程实践中，共遇到两个关键工程问题并均给出了解决方案：

**问题一（ONNX格式）：** YOLOv5简化单输出格式与RKNN量化工具链不兼容，导致INT8量化后检测率归零。解决方案为改用3-branch分离输出格式，将后处理移至C++推理端实现。

**问题二（量化算法）：** EIoU训练模型的激活值分布与Normal（MinMax）量化不适配，导致板端碎框严重。解决方案为改用KL散度量化算法，有效保留激活值分布主体信息，量化精度恢复正常。

这两个工程问题均非文献中常见的算法层面问题，而是工具链对接和训练方法影响量化行为的实践发现，对后续从事RKNN平台部署的工作具有直接的参考价值。

---

# 第六章　可视化管理系统

## 6.1　系统功能概述

在完成检测、跟踪与部署工作的基础上，本项目还集成了一套基于PyQt5开发的可视化管理系统（`gui/` 目录），用于直观展示检测与跟踪结果，辅助系统调试与演示。

该系统提供图像/视频输入选择、模型文件加载、置信度阈值调节、检测结果实时显示（边界框叠加、类别标签）、跟踪ID可视化（轨迹颜色区分）等基础功能。界面由AI工具辅助生成，实现了基本的功能组合，满足演示需求。

可视化系统设计为独立模块，通过调用 `src/detection/` 的检测器接口和 `src/tracking/` 的跟踪器接口实现功能整合，主要服务于系统功能演示，不影响核心算法的研究结论。

---

# 第七章　总结与展望

## 7.1　工作总结

本文以FLIR数据集的红外灰度图像为研究对象，以YOLOv5s为基线框架，围绕轻量化检测模型改进、多目标跟踪算法对比和RV1126B嵌入式端部署三条主线，完成了以下工作：

**（1）系统性消融实验研究。** 设计并执行了两阶段共13组控制变量消融实验，在统一训练口径约束下，定量评估了Ghost-C3、Shuffle-C3、SE注意力、坐标注意力、SIoU损失、EIoU损失六种改进策略在红外灰度图像场景下的有效性。Stage1实验揭示EIoU损失是本场景下性价比最高的单一改进，mAP@0.5较基线提升1.1个百分点；Stage2组合实验验证了EIoU与注意力机制的正向协同效应，并确定了速度-精度折衷方案（Ghost-C3+EIoU）。

**（2）多目标跟踪算法评估。** 以最优检测配置（EIoU）为前端，对DeepSORT、ByteTrack、CenterTrack三种算法进行系统对比，结果表明ByteTrack在红外灰度场景下的MOTA（0.714）、IDF1（0.703）和ID Switch（26次）均优于其他两种算法，分析表明ByteTrack不依赖外观特征的设计规避了灰度图像ReID能力弱的问题，其两轮匹配策略有效缓解了低对比度场景下的目标丢失。

**（3）RV1126B嵌入式部署工程实践。** 完成从PyTorch权重到RKNN INT8量化模型的完整转换链路，设计并实现了高效的C++推理框架，板端全流程推理延迟均值为19.8ms，满足实时性目标。记录并解决了两个关键工程问题：3-branch ONNX格式解决量化后检测失效问题；KL散度量化算法解决EIoU模型与Normal量化不适配的碎框问题。

本文的研究价值在于：首先，在方法论上提供了一套可复现的两阶段控制变量消融框架，研究结论具有明确的可追溯性；其次，在场景上系统覆盖了近红外灰度图像这一在已有文献中缺乏完整对比记录的场景；再者，在工程上形成了从训练到嵌入式部署的完整链路，并记录了量化过程中的关键工程发现，具有实践参考价值。

## 7.2　研究局限与未来展望

尽管本文工作完成了既定目标，仍存在以下局限性，有待后续研究深化：

**局限一：消融实验缺乏纵向深度优化。** 本文的消融实验定位于"配置筛选研究"——在固定训练口径下比较哪种配置最优，而非在某一配置上进行超参数的深度调参（如学习率、数据增强强度的精细搜索）。这意味着所报告的mAP等指标是"统一口径下的可比结果"，而非各配置在其最优超参数下的潜在上限。后续工作可以选取EIoU配置，在其基础上通过贝叶斯优化或网格搜索进行更系统的超参数寻优，有望进一步提升检测性能上限。

**局限二：数据集规模与多样性受限。** 本文使用的FLIR数据集场景相对集中（主要为道路行人车辆），天气条件和成像距离的多样性有限。将研究结论推广至更复杂场景（雨雾天、远距离小目标、密集遮挡）的泛化能力有待验证。

**局限三：跟踪评估场景有限。** 跟踪实验基于FLIR验证集视频序列，未专门构建遮挡、快速运动等极端场景的测试基准。在更严苛的跟踪条件下，算法排序是否稳定有待进一步验证。

**展望一：基于知识蒸馏的模型压缩。** 当前部署采用结构化INT8量化，未来可探索以标准YOLOv5m/l为教师模型、YOLOv5s为学生模型进行知识蒸馏，在保持轻量化结构的前提下提升学生模型的检测精度，进一步缩小量化前后的精度差距。

**展望二：检测-跟踪联合优化。** 当前框架中检测器与跟踪器相互独立，未来可探索以跟踪目标的时序上下文信息反向增强检测器（如通过轨迹预测指导检测区域选择），实现检测与跟踪的深度协同。

**展望三：部署量化精度提升。** 当前KL散度INT8量化引入约1.4个百分点的精度损失，未来可探索混合精度量化（INT8与INT16混合）策略，对精度敏感层保留更高比特精度，在延迟与精度之间寻找更优的平衡点。在RKNN-Toolkit2持续更新的背景下，MMSE量化算法的效率有望改善，届时可在可接受的时间代价内验证其对EIoU模型的量化精度提升效果。

**展望四：多模态融合扩展。** FLIR数据集本身包含同步可见光图像，未来可探索红外与可见光的特征级融合检测框架，充分利用两种模态的互补信息（可见光提供颜色/纹理语义，红外灰度提供形状/轮廓信息），提升目标检测在复杂光照条件下的鲁棒性。

---

# 参考文献

[1] Redmon J, Divvala S, Girshick R, et al. You only look once: Unified, real-time object detection[C]. Proceedings of the IEEE conference on computer vision and pattern recognition, 2016: 779-788.

[2] Redmon J, Farhadi A. YOLOv3: An incremental improvement[J]. arXiv preprint arXiv:1804.02767, 2018.

[3] Bochkovskiy A, Wang C Y, Liao H Y M. YOLOv4: Optimal speed and accuracy of object detection[J]. arXiv preprint arXiv:2004.10934, 2020.

[4] Jocher G, Stoken A, Borovec J, et al. Ultralytics/yolov5: v7.0-YOLOv5 SOTA Realtime Instance Segmentation[EB/OL]. https://github.com/ultralytics/yolov5, 2022.

[5] Liu W, Anguelov D, Erhan D, et al. SSD: Single shot multibox detector[C]. European conference on computer vision, Springer, Cham, 2016: 21-37.

[6] Girshick R. Fast R-CNN[C]. Proceedings of the IEEE international conference on computer vision, 2015: 1440-1448.

[7] Ren S, He K, Girshick R, et al. Faster R-CNN: Towards real-time object detection with region proposal networks[J]. IEEE transactions on pattern analysis and machine intelligence, 2016, 39(6): 1137-1149.

[8] Han K, Wang Y, Tian Q, et al. GhostNet: More features from cheap operations[C]. Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020: 1580-1589.

[9] Ma N, Zhang X, Zheng H T, et al. ShuffleNet V2: Practical guidelines for efficient CNN architecture design[C]. Proceedings of the European conference on computer vision, 2018: 116-131.

[10] Hu J, Shen L, Sun G. Squeeze-and-excitation networks[C]. Proceedings of the IEEE conference on computer vision and pattern recognition, 2018: 7132-7141.

[11] Hou Q, Zhou D, Feng J. Coordinate attention for efficient mobile network design[C]. Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021: 13713-13722.

[12] Rezatofighi H, Tsoi N, Gwak J, et al. Generalized intersection over union: A metric and a loss for bounding box regression[C]. Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019: 658-666.

[13] Zheng Z, Wang P, Liu W, et al. Distance-IoU loss: Faster and better learning for bounding box regression[C]. Proceedings of the AAAI conference on artificial intelligence, 2020, 34(7): 12993-13000.

[14] Zhang Y F, Ren W, Zhang Z, et al. Focal and efficient IOU loss for accurate bounding box regression[J]. Neurocomputing, 2022, 506: 146-157.

[15] Gevorgyan Z. SIoU loss: More powerful learning for bounding box regression[J]. arXiv preprint arXiv:2205.12740, 2022.

[16] Bewley A, Ge Z, Ott L, et al. Simple online and realtime tracking[C]. 2016 IEEE international conference on image processing, 2016: 3464-3468.

[17] Wojke N, Bewley A, Paulus D. Simple online and realtime tracking with a deep association metric[C]. 2017 IEEE international conference on image processing, 2017: 3645-3649.

[18] Zhang Y, Sun P, Jiang Y, et al. ByteTrack: Multi-object tracking by associating every detection box[C]. European conference on computer vision, Springer, Cham, 2022: 1-21.

[19] Zhou X, Koltun V, Krähenbühl P. Tracking objects as points[C]. European conference on computer vision, Springer, Cham, 2020: 474-490.

[20] Gade R, Moeslund T B. Thermal cameras and applications: A survey[J]. Machine vision and applications, 2014, 25(1): 245-262.

[21] Jacob B, Kligys S, Chen B, et al. Quantization and training of neural networks for efficient integer-arithmetic-only inference[C]. Proceedings of the IEEE conference on computer vision and pattern recognition, 2018: 2704-2713.

[22] Migacz S. 8-bit inference with TensorRT[C]. GPU technology conference, 2017.

[23] FLIR Systems. FLIR thermal dataset for algorithm training[EB/OL]. https://www.flir.com/oem/adas/adas-dataset-form/, 2019.

[24] He K, Zhang X, Ren S, et al. Deep residual learning for image recognition[C]. Proceedings of the IEEE conference on computer vision and pattern recognition, 2016: 770-778.

[25] Lin T Y, Dollar P, Girshick R, et al. Feature pyramid networks for object detection[C]. Proceedings of the IEEE conference on computer vision and pattern recognition, 2017: 2117-2125.

[26] Liu S, Qi L, Qin H, et al. Path aggregation network for instance segmentation[C]. Proceedings of the IEEE conference on computer vision and pattern recognition, 2018: 8759-8768.

[27] Bernardin K, Stiefelhagen R. Evaluating multiple object tracking performance: The CLEAR MOT metrics[J]. EURASIP Journal on Image and Video Processing, 2008, 2008: 1-10.

[28] Ristani E, Solera F, Zou R, et al. Performance measures and a data set for multi-target, multi-camera tracking[C]. European conference on computer vision, Springer, Cham, 2016: 17-35.

[29] Rokchip Electronics. RV1126 Technical Reference Manual[EB/OL]. https://www.rock-chips.com/a/cn/product/RV11xx_Series/2020/0211/1168.html, 2020.

[30] Rockchip. RKNN-Toolkit2 User Guide[EB/OL]. https://github.com/rockchip-linux/rknn-toolkit2, 2023.

---

# 致谢

本论文的完成离不开多方面的支持与帮助。首先，衷心感谢指导教师张建锋老师在课题选题、研究方向确定及论文写作全程中给予的悉心指导。张老师在中期汇报中提出的"消融实验应兼顾纵向深化"的建议，促使本文在消融实验结论阐释和研究边界声明上做了更为严谨的表述，对论文质量的提升有重要意义。

感谢信息工程学院为本科毕业设计提供的实验计算资源支持，以及同学们在调试过程中的交流帮助。感谢FLIR公司公开发布的多模态红外数据集，以及Ultralytics、Rockchip等开源社区为本研究提供的工具链支持。

最后，感谢家人的理解与支持，为本人顺利完成四年本科学习和毕业论文写作提供了坚实的后盾。


---

# 附录A　消融实验完整配置清单

## A.1　Stage1实验配置明细

本附录记录13组消融实验的完整配置，以供结果可复现性验证。所有实验均使用 `scripts/train/train_ablation.py --profile controlled` 执行，训练口径统一由 `configs/ablation/train_profile_controlled.yaml` 约束。

**Exp1（基线YOLOv5s）**
- 模型配置文件：`model/yolov5/configs/yolov5s_base.yaml`
- 超参数配置：YOLOv5默认超参数（无覆盖）
- 训练命令要点：epochs=100, batch=16, img=640, cos_lr=True, patience=20
- 输出目录：`outputs/ablation_study/ablation_exp01_baseline/`
- 说明：以YOLOv5s原始架构和默认CIoU损失作为所有对比实验的基准参考点

**Exp2（Ghost-C3）**
- 模型配置文件：`model/yolov5/configs/yolov5s_lightweight.yaml`
- 超参数配置：无覆盖（使用默认CIoU损失）
- 改动要点：将Backbone中所有C3模块替换为GhostC3模块，Neck中C3模块同步替换
- 输出目录：`outputs/ablation_study/ablation_exp02_ghost/`
- 说明：评估Ghost卷积在红外灰度场景下的独立轻量化效果

**Exp3（Shuffle-C3）**
- 模型配置文件：`model/yolov5/configs/yolov5s_shuffle.yaml`
- 超参数配置：无覆盖（使用默认CIoU损失）
- 改动要点：将Backbone中C3模块替换为基于ShuffleNet分组卷积的Shuffle-C3模块
- 输出目录：`outputs/ablation_study/ablation_exp03_shuffle/`
- 说明：评估通道混洗机制在红外灰度场景下的独立效果

**Exp4（SE注意力）**
- 模型配置文件：`model/yolov5/configs/yolov5s_attention.yaml`
- 超参数配置：无覆盖（使用默认CIoU损失）
- 改动要点：在Backbone各Stage后插入SE（Squeeze-and-Excitation）注意力模块，通道数缩放比r=16
- 输出目录：`outputs/ablation_study/ablation_exp04_attention/`
- 说明：评估通道注意力机制对红外目标特征重标定的独立效果

**Exp5（坐标注意力）**
- 模型配置文件：`model/yolov5/configs/yolov5s_coordatt.yaml`
- 超参数配置：无覆盖（使用默认CIoU损失）
- 改动要点：在Backbone各Stage后插入CoordAttention模块，嵌入位置信息的通道注意力
- 输出目录：`outputs/ablation_study/ablation_exp05_coordatt/`
- 说明：评估含位置信息的注意力机制对目标定位的辅助效果

**Exp6（SIoU损失）**
- 模型配置文件：`model/yolov5/configs/yolov5s_base.yaml`（与基线相同架构）
- 超参数配置：`configs/ablation/hyp_siou_only.yaml`（仅修改损失函数为SIoU）
- 改动要点：将YOLOv5默认CIoU边界框回归损失替换为SIoU损失
- 输出目录：`outputs/ablation_study/ablation_exp06_siou/`
- 说明：评估角度感知损失函数在行人/车辆矩形目标场景下的独立效果

**Exp7（EIoU损失）**
- 模型配置文件：`model/yolov5/configs/yolov5s_base.yaml`（与基线相同架构）
- 超参数配置：`configs/ablation/hyp_eiou_only.yaml`（仅修改损失函数为EIoU）
- 改动要点：将YOLOv5默认CIoU边界框回归损失替换为EIoU损失
- 输出目录：`outputs/ablation_study/ablation_exp07_eiou/`
- 说明：评估宽高分离约束损失函数的独立提升效果，Stage1最优配置

## A.2　Stage2实验配置明细

**Exp8（Ghost-C3 + SE注意力）**
- 模型配置文件：`model/yolov5/configs/yolov5s_ghost_attention.yaml`
- 超参数配置：无覆盖（使用默认CIoU损失）
- 改动要点：在Ghost-C3轻量化架构基础上添加SE注意力模块
- 输出目录：`outputs/ablation_study/ablation_exp08_ghost_attention/`
- 说明：探究轻量化结构与注意力机制在无损失函数改进时的组合效果

**Exp9（Ghost-C3 + EIoU）**
- 模型配置文件：`model/yolov5/configs/yolov5s_lightweight.yaml`
- 超参数配置：`configs/ablation/hyp_eiou_only.yaml`
- 改动要点：Ghost-C3轻量化架构配合EIoU损失函数
- 输出目录：`outputs/ablation_study/ablation_exp09_ghost_eiou/`
- 说明：探究轻量化+EIoU的速度-精度折衷方案

**Exp10（SE注意力 + EIoU）**
- 模型配置文件：`model/yolov5/configs/yolov5s_attention.yaml`
- 超参数配置：`configs/ablation/hyp_eiou_only.yaml`
- 改动要点：SE注意力模块配合EIoU损失函数，架构不做轻量化
- 输出目录：`outputs/ablation_study/ablation_exp10_attention_eiou/`
- 说明：Stage2精度最优配置，验证注意力+EIoU的正向协同

**Exp11（Shuffle-C3 + 坐标注意力）**
- 模型配置文件：`model/yolov5/configs/yolov5s_shuffle_coordatt.yaml`
- 超参数配置：无覆盖（使用默认CIoU损失）
- 改动要点：Shuffle-C3轻量化架构配合坐标注意力
- 输出目录：`outputs/ablation_study/ablation_exp11_shuffle_coordatt/`
- 说明：探究两种轻量化相关改进的组合效果（无损失函数改进）

**Exp12（Shuffle-C3 + 坐标注意力 + SIoU）**
- 模型配置文件：`model/yolov5/configs/yolov5s_shuffle_coordatt.yaml`
- 超参数配置：`configs/ablation/hyp_siou_only.yaml`
- 改动要点：在Exp11基础上引入SIoU损失函数
- 输出目录：`outputs/ablation_study/ablation_exp12_shuffle_coordatt_siou/`
- 说明：验证SIoU在三组合场景下的持续无效性

**Exp13（Shuffle-C3 + 坐标注意力 + EIoU）**
- 模型配置文件：`model/yolov5/configs/yolov5s_shuffle_coordatt.yaml`
- 超参数配置：`configs/ablation/hyp_eiou_only.yaml`
- 改动要点：在Exp11基础上引入EIoU损失函数，三者组合
- 输出目录：`outputs/ablation_study/ablation_exp13_shuffle_coordatt_eiou/`
- 说明：探究Shuffle-C3+坐标注意力+EIoU的三方组合效果，Stage2速度折衷方案

## A.3　超参数文件内容摘要

`configs/ablation/hyp_eiou_only.yaml` 的核心改动为将损失函数类型标记为 `eiou`，其余学习率、动量、权重衰减、数据增强参数均与YOLOv5s默认超参数保持一致（不做额外调整），以确保"仅损失函数不同"的控变量要求。

`configs/ablation/hyp_siou_only.yaml` 的核心改动类似，仅将损失函数类型标记为 `siou`，其余参数不变。

这种"最小改动"的超参数文件设计策略，使得实验间的参数差异可以精确追溯，避免了因多参数同时变化导致的归因困难。


---

# 附录B　部署工程文档

## B.1　PC端环境配置

部署链路的PC端工作在Ubuntu 20.04 LTS操作系统下进行，Python版本3.8，PyTorch版本1.12.0，ONNX版本1.13.0，RKNN-Toolkit2版本1.5.2。交叉编译工具链使用 `arm-linux-gnueabihf-g++ 9.4.0`，OpenCV版本4.5.5（ARM交叉编译版）。

主要依赖安装命令（记录如下以供复现）：

```bash
# Python环境
pip install rknn-toolkit2==1.5.2
pip install onnxruntime==1.13.0
pip install torch==1.12.0 torchvision==0.13.0

# 交叉编译工具链
sudo apt install gcc-arm-linux-gnueabihf g++-arm-linux-gnueabihf
```

RKNN-Toolkit2的安装需要根据目标平台（rv1126）选择对应的whl包，与针对RK3588等其他平台的安装包不同，使用时需注意平台匹配。

## B.2　ONNX导出与验证流程

完整ONNX导出脚本调用链如下：

```bash
# 1. 从PyTorch权重导出ONNX（3-branch输出格式）
cd /path/to/yolov5
python export.py \
    --weights ../outputs/ablation_study/ablation_exp07_eiou/weights/best.pt \
    --include onnx \
    --opset 12 \
    --img 640 \
    --batch 1 \
    --simplify  # 注意：simplify在3-branch格式下仍可使用，不会合并输出头

# 2. 验证ONNX输出格式
python -c "
import onnxruntime as ort
sess = ort.InferenceSession('best.onnx')
print('输入:', [i.name for i in sess.get_inputs()])
print('输出:', [o.name for o in sess.get_outputs()])
print('输出shape:', [o.shape for o in sess.get_outputs()])
"
```

若输出显示三个output节点（output0/output1/output2），则格式正确；若显示单一output节点，需检查YOLOv5导出代码中的 `export_formats` 和 `--opset` 设置。

## B.3　RKNN量化转换脚本结构

量化转换脚本位于 `deploy/rv1126b_yolov5/python/` 目录，主要结构如下：

```
python/
├── convert_rknn.py      # 主转换脚本
├── verify_rknn.py       # PC端仿真验证脚本
└── gen_calibration.py   # 生成校准集文件列表脚本
```

`convert_rknn.py` 的主要步骤：
1. 初始化RKNN对象，配置目标平台、量化算法、归一化参数
2. 加载ONNX模型，指定三个输出节点名称
3. 构建量化模型（`rknn.build(do_quantization=True, dataset=...)`）
4. 导出.rknn文件（`rknn.export_rknn(...)`）
5. 在PC端仿真器上运行一张测试图，验证输出形状正确、无NaN/Inf

`verify_rknn.py` 在PC端使用RKNN仿真器（`rknn.init_runtime(target=None)`）加载量化后的.rknn模型，对验证集中的若干图像执行推理，计算与FP32推理结果的mAP差异，用于量化精度快速评估（无需实际板端）。

## B.4　C++工程文件结构

```
deploy/rv1126b_yolov5/
├── CMakeLists.txt              # 交叉编译CMake配置
├── build_rv1126b.sh            # 一键交叉编译脚本
├── calibration_dataset.txt     # 量化校准集图像路径列表
├── model/
│   └── yolov5s_eiou_int8_kl.rknn  # 量化后模型文件
├── python/                     # PC端转换与验证脚本
├── scripts/                    # 板端测试脚本
└── src/
    ├── main.cpp                # 程序入口（命令行参数解析）
    ├── detector.h/.cpp         # 检测器类（封装RKNN推理接口）
    ├── preprocess.h/.cpp       # 图像预处理（letterbox、归一化）
    └── postprocess.h/.cpp      # 后处理（解码、NMS）
```

`detector.h` 定义的核心接口：
```cpp
class YoloDetector {
public:
    YoloDetector(const std::string& model_path, int img_size = 640);
    ~YoloDetector();
    std::vector<Detection> detect(const cv::Mat& frame, float conf_thresh = 0.25f, float nms_thresh = 0.45f);
private:
    rknn_context ctx_;
    int img_size_;
    std::vector<rknn_tensor_attr> output_attrs_;
};
```

## B.5　板端测速脚本

板端测速通过以下Shell脚本执行（`deploy/rv1126b_yolov5/scripts/benchmark.sh`）：

```bash
#!/bin/bash
# 板端测速脚本
IMG_DIR=/data/flir_val_images
MODEL=/data/yolov5s_eiou_int8_kl.rknn
BINARY=/data/yolo_detector

echo "开始板端推理性能测试..."
$BINARY --model $MODEL --img-dir $IMG_DIR --n 100 --warmup 10 \
    --output /data/benchmark_results.json

echo "测试完成，结果已保存至 /data/benchmark_results.json"
```

其中 `--warmup 10` 表示前10次推理为预热阶段，不计入统计；`--n 100` 表示统计100次有效推理的延迟分布。测速结果以JSON格式记录，包含每次推理的延迟分解（预处理/NPU推理/后处理）。


---

# 附录C　模型结构详细说明

## C.1　YOLOv5s网络结构详述

YOLOv5s的完整网络结构由Backbone、Neck和Head三部分组成，各部分的层级结构及通道数配置如下。

**Backbone（CSPDarknet53-Small变体）**

输入图像尺寸为640×640×3（灰度图复制为3通道）。Backbone通过以下顺序的模块进行特征提取：

- 第一层：Focus模块（将输入划分为4个子采样图后拼接，640×640→320×320，输出通道32）——在YOLOv5后续版本中被标准6×6卷积替代，功能等价；
- P2阶段：标准卷积（320×320→160×160，通道64）+ C3模块（1×Bottleneck，通道64）；
- P3阶段：标准卷积（160×160→80×80，通道128）+ C3模块（3×Bottleneck，通道128）；
- P4阶段：标准卷积（80×80→40×40，通道256）+ C3模块（3×Bottleneck，通道256）；
- P5阶段：标准卷积（40×40→20×20，通道512）+ C3模块（1×Bottleneck，通道512）+ SPPF模块（空间金字塔池化，通道512）。

其中C3模块的核心为交叉阶段部分连接（CSP）结构：输入特征图经过1×1卷积分支后进入Bottleneck堆叠，另一路直接经过1×1卷积，两路在通道维度拼接后再经1×1卷积输出，这种设计在减少计算量的同时保留了梯度多路径流动的特性。SPPF（Spatial Pyramid Pooling Fast）模块对特征图进行多尺度池化（kernel 5×5，串联3次），在固定输出尺寸的同时扩大感受野。

**Neck（PANet）**

Neck以PANet结构融合Backbone输出的三个尺度特征（P3/P4/P5）：

- 自顶向下FPN通路：P5上采样（最近邻插值）至40×40后与P4拼接→C3处理；P4上采样至80×80后与P3拼接→C3处理；生成富含语义信息的大中小三尺度特征图；
- 自底向上PA通路：80×80特征经标准卷积下采样至40×40后与FPN输出的40×40特征拼接→C3处理；40×40特征再次下采样至20×20后与P5拼接→C3处理；生成富含定位信息的三尺度特征图。

最终Neck输出三个不同分辨率的融合特征图：80×80（对应小目标，步幅8）、40×40（中目标，步幅16）、20×20（大目标，步幅32）。

**Head（多尺度检测头）**

三个检测头分别对三个特征图进行预测，每个检测头对每个空间位置预测3个锚框（anchor），每个锚框输出的预测向量维度为 $5 + \text{nc} = 5 + 2 = 7$（行人/车辆2类）：

$$\text{pred} = [t_x, t_y, t_w, t_h, \text{obj\_conf}, p_{\text{person}}, p_{\text{car}}]$$

其中 $(t_x, t_y)$ 为相对当前格子左上角的中心点偏移（sigmoid激活后限定在0到1之间），$(t_w, t_h)$ 为相对对应锚框宽高的对数偏移，$\text{obj\_conf}$ 为目标存在置信度（sigmoid激活），$p_c$ 为各类别条件概率（sigmoid激活，多标签分类）。

锚框尺寸（在640×640输入下）：
- 小目标检测头（80×80）：(10,13), (16,30), (33,23)
- 中目标检测头（40×40）：(30,61), (62,45), (59,119)
- 大目标检测头（20×20）：(116,90), (156,198), (373,326)

## C.2　Ghost-C3模块结构

GhostC3模块用GhostBottleneck替代C3中的标准Bottleneck，GhostBottleneck的结构为：

1. 第一Ghost卷积：以标准卷积（1×1或3×3）生成1/2数量的"本征特征图"，再通过深度可分离卷积（DW Conv，1×1）生成剩余1/2的"幻象特征图"，两者拼接得到完整的中间特征图；
2. 深度可分离卷积（中间层，用于维度变换）；
3. 第二Ghost卷积（输出层）。

相比标准Bottleneck，GhostBottleneck的FLOPs减少约为：假设输入通道c，Ghost比例 $\lambda = 0.5$，则计算量约为标准Bottleneck的 $(1 + 1/s^2)/(2)$ 倍，其中s为深度卷积的kernel size。在YOLOv5s规模下，使用Ghost-C3后Backbone的FLOPs约降低35%。

## C.3　坐标注意力模块结构

CoordAttention模块的计算流程如下：

**水平/垂直方向特征嵌入：** 对于输入特征图 $X \in \mathbb{R}^{C \times H \times W}$，分别执行水平和垂直方向的全局平均池化：

$$z_c^h(h) = \frac{1}{W} \sum_{0 \le i < W} X_c(h, i), \quad z_c^v(v) = \frac{1}{H} \sum_{0 \le j < H} X_c(j, v)$$

将 $z^h \in \mathbb{R}^{C \times H \times 1}$ 和 $z^v \in \mathbb{R}^{C \times 1 \times W}$ 拼接为 $f \in \mathbb{R}^{C \times 1 \times (H+W)}$。

**特征变换：** 经过1×1卷积将通道数降至 $C/r$（r=32为压缩比），再通过批归一化和非线性激活（hard-swish）得到共享特征张量，最后分别经过两个1×1卷积还原至C通道，得到水平注意力权重 $a^h \in \mathbb{R}^{C \times H \times 1}$ 和垂直注意力权重 $a^v \in \mathbb{R}^{C \times 1 \times W}$（sigmoid激活归一化）。

**特征重标定：** 输出 $Y_c(h, w) = X_c(h, w) \cdot a_c^h(h) \cdot a_c^v(w)$，即在通道、水平、垂直三个维度同时施加注意力权重。

与SE注意力相比，CoordAttention的参数量略高（因为同时处理H和W维度），但能为每个通道分配不同的空间位置权重，而SE的通道注意力与空间位置无关，理论上对于位置敏感的目标定位任务更具优势。


---

# 补充内容：各章节详细展开

## 补充2.7　训练策略深度解析

YOLOv5的训练策略是保证消融实验对比公平性的重要基础，以下对关键训练技术进行详细说明。

**马赛克数据增强（Mosaic Augmentation）** 是YOLOv5中最重要的数据增强策略之一。其原理为：在每次训练迭代中，随机从数据集中选取4张图像，将其随机缩放后拼接为一张训练图（每张图像占据输出图像的一个象限，拼接点随机选取）。这一策略的优点在于：（1）每张训练图同时包含4个不同场景的目标，大幅增加了批次内目标的多样性；（2）将多张图像压缩至更小区域，等效提升了小目标的比例，改善了对小目标的检测能力；（3）使模型见到更多拼接边界处的截断目标，提升了对部分遮挡目标的鲁棒性。在红外灰度图像场景下，马赛克增强同样有效，因为其核心价值在于样本多样性而非颜色信息。

**混合精度训练（AMP，Automatic Mixed Precision）** 允许训练过程中部分计算在FP16精度下执行，以减少显存占用和加速矩阵运算，同时通过梯度缩放（Gradient Scaling）保证数值稳定性。本文所有实验均启用AMP，在保证训练稳定性的前提下提升了每个实验的训练速度。

**余弦退火学习率调度（Cosine LR Annealing）** 使学习率在整个训练过程中按余弦函数从初始值（本实验初始lr=0.01）平滑衰减至最终值（lrf=0.01×0.01=0.0001），避免了阶梯式衰减在切换点处引入的训练不稳定性。余弦退火的数学表达为：

$$\text{lr}_t = \text{lr}_f + \frac{1}{2}(\text{lr}_0 - \text{lr}_f)\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)$$

其中 $t$ 为当前训练轮次，$T$ 为总轮次（100），$\text{lr}_0$ 为初始学习率，$\text{lr}_f$ 为最终学习率。

**早停机制（Early Stopping）** 监测验证集的mAP@0.5指标，当连续patience=20轮验证集mAP未改善时终止训练。这一机制防止了模型过拟合——在验证集mAP不再提升后继续训练只会导致验证损失上升、泛化能力下降。在本实验中，大多数实验在70～90轮左右触发早停，实际训练轮次因实验配置不同而略有差异，但均未超过100轮上限。

**标签平滑（Label Smoothing）** 在本文的消融实验中未启用（`label_smoothing: null`），以避免引入额外变量。标签平滑将硬标签（0或1）替换为软标签（如 $\epsilon=0.1$ 时为0.1或0.9），能在一定程度上防止分类过拟合，但对边界框回归损失（EIoU/SIoU/CIoU）的影响不显著，因此在以边界框回归为核心关注点的消融实验中，其引入会增加变量复杂度而不带来明显收益。

## 补充3.6　训练过程分析

**损失曲线特征分析**

在13组消融实验中，训练损失和验证损失曲线的整体形态呈现出一致的下降趋势，但不同实验在收敛速度和最终损失值上存在可观察的差异。

EIoU实验（Exp7）的边界框回归损失（box loss）在前30轮的下降速率明显快于基线（Exp1），这与EIoU损失函数的梯度特性有关：EIoU对宽高误差的独立约束使得边界框在宽度和高度两个方向上可以同时独立地以较大梯度更新，而CIoU的联合约束在宽高比接近目标时梯度趋于饱和，收敛减慢。

SIoU实验（Exp6）在前20轮的训练损失下降存在轻微波动（相比基线），推测与SIoU角度损失项在初期框未对齐时梯度较大、导致早期训练不稳定有关。这一现象在后期（30轮后）消失，训练曲线趋于平稳，但最终收敛的验证集mAP略低于基线，说明SIoU的角度损失项对本场景目标（形状规整的行人和车辆）引入了不必要的约束复杂度。

Ghost-C3实验（Exp2和Exp3）的训练速度（每轮时间）因参数量减少而略快（约快10%），但最终验证集mAP偏低，说明精度-速度的折衷在训练阶段已经体现：Ghost近似特征的表达能力上限低于标准C3特征。

**验证集精确率-召回率平衡分析**

各实验在验证集上的PR曲线形态分析：EIoU实验（Exp7）在高精确率区间（P>0.8）的召回率明显高于基线，说明EIoU改善了边界框的定位质量，使得在严格IoU阈值下（0.5:0.95区间）的匹配率更高，这直接体现为mAP@0.5:0.95的显著提升（+1.9%）。而在较低的IoU阈值（0.5）下，各实验的差距相对较小，说明EIoU的主要贡献在于精细定位而非粗略检测率的提升。

轻量化结构实验（Exp2、Exp3）在高召回率区间（R>0.7）的精确率明显低于基线，说明这两种配置在保证覆盖更多目标时，引入了更多假阳性检测，检测质量略有下降。

注意力机制实验（Exp4、Exp5）的PR曲线形态与基线相似，但整体向右上方轻微偏移，说明注意力机制的贡献是均匀地提升了检测质量（精确率和召回率均有小幅改善），而非在某一指标上的侧重优化。

## 补充3.7　类别分析：行人vs车辆检测性能

本文以行人（person）和车辆（car）为检测类别，两类目标在红外灰度图像中的成像特性和检测难度存在明显差异，有必要对两类别分别分析。

**行人检测性能分析**

行人目标在FLIR灰度红外图像中通常呈现较窄的竖直矩形，平均宽高比约为1:3。在近距离（目标较大）时，行人的体态轮廓清晰，检测相对容易；在中远距离（目标较小，边界框约为20×60像素）时，行人目标与背景的对比度降低，检测难度大幅上升。

EIoU损失对行人检测的提升尤为明显（mAP提升约+1.5%），主要原因是：行人的高度方向（纵向）误差和宽度方向（横向）误差通常呈现不同的分布特性——高度误差受姿态变化影响较大，宽度误差受服装厚度变化影响较大。EIoU的宽高分离约束使网络能够分别针对两个维度的误差进行独立优化，而CIoU的联合宽高比约束在这种非对称误差场景下的训练效果受限。

**车辆检测性能分析**

车辆目标通常呈现较宽的横向矩形，俯视角度下（道路行车视角）宽高比约为2:1，侧面视角下约为3:1。车辆目标整体尺寸较大，在640×640输入下边界框面积通常远大于行人，属于中大目标范畴，检测召回率相对较高。

但车辆检测面临的主要困难是：多辆车辆并排停放时，相邻车辆的边界框高度重叠，NMS过程中存在误抑制风险（两辆相邻车辆的IoU超过NMS阈值时，其中一辆可能被错误过滤）。ByteTrack在跟踪阶段的两轮匹配策略也有助于缓解这一问题：被NMS过滤的低置信度车辆检测框可能在跟踪的第二轮匹配中被找回，保持轨迹连续性。

在所有13组实验中，车辆类别的mAP@0.5普遍高于行人（约高5～8个百分点），说明当前数据集和模型配置下，车辆检测整体较为稳定，行人检测是决定整体mAP的主要瓶颈，也是各改进策略效果分化的主要来源。

## 补充4.6　跟踪系统工程实现细节

**轨迹生命周期管理**

在跟踪系统的实现中，轨迹的生命周期由以下三个状态管理：

- **Tentative（候选状态）**：新检测到的目标初始化为候选轨迹，连续出现n_init帧（默认3帧）后升级为Confirmed状态；
- **Confirmed（已确认状态）**：活跃跟踪中的轨迹，持续与新检测结果匹配更新；
- **Deleted（已终止状态）**：连续max_age帧未匹配成功的轨迹，从活跃列表中移除，ID不再复用。

这一三状态设计的目的是：Tentative状态过滤掉单帧出现的虚假检测（如背景噪声产生的单帧错误框），防止大量临时轨迹被记录为正式目标；Deleted状态的延迟终止（而非立即删除）为短时遮挡提供了恢复窗口。

**卡尔曼滤波状态向量设计**

本文实现的卡尔曼滤波器使用8维状态向量：

$$\mathbf{x} = [c_x, c_y, s, r, \dot{c}_x, \dot{c}_y, \dot{s}, \dot{r}]^T$$

其中 $c_x, c_y$ 为边界框中心坐标，$s$ 为边界框面积，$r$ 为宽高比，$\dot{c}_x, \dot{c}_y, \dot{s}, \dot{r}$ 为对应量的一阶速度估计。观测向量为4维 $[c_x, c_y, s, r]^T$。状态转移矩阵假设匀速运动模型：位置在下一帧的预测值为当前位置加上速度与帧间隔的乘积。过程噪声协方差和观测噪声协方差通过经验设定，对于速度项的不确定性给予较大的过程噪声（允许速度快速变化），对于位置项的测量给予较小的观测噪声（信任检测器输出的位置）。

**IoU代价矩阵计算**

在ByteTrack和SORT中，代价矩阵基于IoU距离计算：

$$d_{ij} = 1 - \text{IoU}(\text{track}_i^{\text{pred}}, \text{det}_j)$$

其中 $\text{track}_i^{\text{pred}}$ 为卡尔曼滤波预测的轨迹i在当前帧的位置，$\text{det}_j$ 为第j个检测框。当 $d_{ij} > 1 - \text{IoU\_thresh}$（即IoU低于阈值，通常0.3）时，该配对直接标记为不可匹配（infinite cost），不参与匈牙利匹配。这一门限设计的目的是：如果预测位置与检测框的IoU极小，说明两者很可能不属于同一目标（尤其是在目标密集或快速运动场景），强制阻止这类低质量匹配，避免引入错误的ID关联。

## 补充5.7　量化过程调试记录

在完成RKNN量化部署的过程中，经历了若干调试阶段，以下记录主要调试步骤和中间发现，作为工程实践参考。

**调试阶段一：确认PC端仿真推理正常**

首先在PC端使用RKNN仿真器（`init_runtime(target=None)`）验证量化模型的基本推理结果。使用Normal量化算法时，仿真推理结果显示置信度分数整体分布异常（大量分数聚集在0.01以下），而KL散度量化后，置信度分数分布恢复至正常范围（有效检测框的置信度集中在0.3～0.9区间）。这一PC端仿真阶段的发现使我们在将模型传输至板端之前，就确认了量化算法选择的正确性，节省了大量的板端调试时间。

**调试阶段二：ONNX格式兼容性验证**

在发现单输出格式ONNX量化失效后，通过对比单输出和3-branch格式的ONNX图结构，确认了问题根源：单输出格式中包含了三个检测头的sigmoid→reshape→concat后处理算子，这些算子在RKNN量化工具链的某个版本下存在量化参数估计的bug（可能与rknn-toolkit2版本1.5.2有关，后续版本可能已修复）。使用onnx模型可视化工具（Netron）对比了两种格式的计算图，确认3-branch格式在检测头的sigmoid层之前截断，RKNN工具链只需量化卷积特征提取部分，规避了后处理算子的量化问题。

**调试阶段三：C++推理输出尺度还原**

在C++推理实现中，后处理阶段需要将NPU输出的检测框坐标从网络输入尺度（640×640）映射回原始图像坐标系。由于预处理采用了letterbox缩放（保持宽高比等比例缩放，不足部分填充灰色边框），还原时需要考虑上下左右的填充量。填充量计算公式如下：

设原始图像宽高为 $(W_0, H_0)$，网络输入尺寸为 $S=640$，则：

$$\text{scale} = \min\left(\frac{S}{W_0}, \frac{S}{H_0}\right)$$

$$\text{pad}_w = \frac{S - W_0 \cdot \text{scale}}{2}, \quad \text{pad}_h = \frac{S - H_0 \cdot \text{scale}}{2}$$

检测框坐标还原：

$$x_0 = \frac{x_{\text{net}} - \text{pad}_w}{\text{scale}}, \quad y_0 = \frac{y_{\text{net}} - \text{pad}_h}{\text{scale}}$$

在C++实现中，$\text{pad}_w$ 和 $\text{pad}_h$ 在预处理阶段计算并传递给后处理模块，确保坐标还原精度。

**调试阶段四：NMS超参数调整**

在板端初步测试时，发现某些场景下检测框密集聚集（多个高度重叠的框）。分析原因：板端推理由于量化误差，置信度分数的分布相比PC端略有偏移，原本被NMS过滤的低质量候选框在量化后置信度略有上升，超过了检测阈值（0.25）但低于NMS阈值所需的IoU（0.45）。通过将检测阈值从0.25提升至0.30，过滤了大部分低质量候选框，检测结果更为干净，同时对高置信度目标的检测率影响微乎其微（高置信度目标的置信度通常大于0.6）。


## 补充1.5　研究意义的延伸阐述

**工程价值层面的进一步说明**

本研究之所以选择RV1126B作为目标部署平台，而非通用GPU服务器，有着明确的应用导向：安防摄像头、低空无人机载端、工厂巡检机器人等实际应用场景，不可能配备高功耗GPU，通常采用ARM+NPU的低功耗嵌入式架构。因此，从"训练完成的检测模型"到"能在嵌入式设备上稳定运行的量化推理程序"之间存在一条工程鸿沟，而现有文献往往只覆盖前半段（训练和评估），对后半段（量化、部署、板端调试）的系统性记录较为稀缺。

本文完整记录了RV1126B部署链路上的两个实质性工程问题及其解决方案，这类"踩坑记录"对于同类嵌入式部署工作具有直接的参考价值——工程师无需重复同样的错误路径，可以直接采用经验证的格式（3-branch ONNX + KL散度量化）作为起点。这是本文工程贡献中比指标数字更具长期价值的部分。

**学术价值层面的补充说明**

关于研究方法论，本文的两阶段消融实验框架可视为一种针对场景适配性的"系统性筛选"研究（Systematic Selection Study）——其目标不是"将某一方法推向极限"，而是"在给定场景约束下，确定哪类方法有效、哪类方法无效"。这类研究在工业界和应用研究领域具有重要的实用价值，因为实际工程选型往往需要此类横向对比基准，而不是单一方法在理想条件下的极限性能。

从另一个角度看，本文13组实验可以视为一种小规模的"AutoML搜索"的手动实现——在预设的搜索空间（{Ghost-C3, Shuffle-C3, SE-Att, CoordAtt} × {默认损失, SIoU, EIoU}的部分空间）内通过枚举评估找到最优配置。相比随机搜索，两阶段设计减少了无效组合的评估数量（Stage1先筛选有效单组件，Stage2再组合）；相比完整网格搜索（全量组合数为6×3=18组），13组实验覆盖了最关键的配置，在资源约束下的搜索效率较优。

## 补充2.8　深度学习目标检测评估体系详述

本节对本文使用的检测评估指标进行更深入的说明，以帮助理解实验结果的解读方式。

**精确率-召回率曲线（PR曲线）的生成方法**

对于每个类别，将验证集上所有检测框按置信度从高到低排序，依次计算在不同置信度阈值下的精确率P和召回率R，得到PR曲线上的一系列点。AP（平均精度）为PR曲线下的面积（积分），通常采用11点插值法或连续积分法计算。mAP为所有类别AP的均值（本文为行人和车辆两类的AP均值）。

**mAP@0.5与mAP@0.5:0.95的区别**

mAP@0.5在计算每个检测框是否为真正例（TP）时，使用固定的IoU阈值0.5：若检测框与真实框的IoU≥0.5，则视为TP，否则为FP。这是早期Pascal VOC数据集采用的评估标准，对定位精度要求相对宽松。

mAP@0.5:0.95采用COCO标准，分别在IoU阈值0.50、0.55、0.60、0.65、0.70、0.75、0.80、0.85、0.90、0.95共10个阈值下计算mAP，再取平均。这10个阈值的平均mAP对边界框定位精度的要求更为严格——要在IoU=0.95的阈值下被计为TP，检测框必须与真实框高度重合，对边界框的定位精度要求接近像素级精确。

因此，mAP@0.5:0.95的提升（尤其是+1.9%）比mAP@0.5的提升（+1.1%）更能说明EIoU损失对边界框定位质量的实质性改善，而非仅仅提高了粗略检测率。

**per-class AP分析的重要性**

在多类别检测中，mAP掩盖了类别间的性能差异。在本文的实验结果中，行人类别的AP通常比车辆类别的AP低5～8个百分点，说明行人检测是性能瓶颈所在。EIoU改进对行人AP的提升（约+1.5%）略高于对车辆AP的提升（约+0.7%），印证了EIoU对宽高比不规则（行人的宽高比变化更大）目标的边界框优化效果更为显著。

## 补充4.7　MOT评估数据集与测评工具详述

**评估数据集描述**

跟踪评估使用FLIR数据集中附带的视频序列作为测试基础。由于FLIR原始数据集以独立帧图像提供而非视频序列格式，本文使用 `scripts/data/` 下的图像序列转视频工具将连续图像帧组织为可供跟踪评估的序列，并手动生成地面真值（Ground Truth）轨迹标注文件（MOT格式：每行包含帧号、目标ID、边界框坐标、置信度等字段）。

评估序列的基本统计：视频帧率约为15FPS（FLIR ADK相机采集帧率），每个评估序列长度约200～500帧，场景类型包括道路行人流场景（行人密集、横向行走多）和停车场/路口场景（车辆密集、遮挡多）。

**py-motmetrics工具使用**

MOT指标计算使用开源的py-motmetrics库（`motmetrics.metrics`），其接受两种格式的输入：地面真值（GT）文件和跟踪结果（HYP）文件，均采用MOT标准CSV格式。评估调用示例：

```python
import motmetrics as mm
acc = mm.MOTAccumulator(auto_id=True)
# 逐帧更新GT与HYP的匹配结果
for frame_id, gt_boxes, hyp_boxes, distances in frame_data:
    acc.update(gt_ids, hyp_ids, distances)
# 计算指标
mh = mm.metrics.create()
summary = mh.compute(acc, metrics=['num_switches', 'mota', 'motp', 'idf1'])
```

距离矩阵（distances）采用1-IoU计算，当IoU低于0.5时（即1-IoU>0.5），对应配对距离设置为 `np.nan`（标记为不可匹配）。

**FPS测量方法**

系统FPS统计的是从视频帧输入到最终跟踪结果输出的端到端处理速度（包含检测器推理+跟踪算法更新），在GPU服务器（NVIDIA RTX 3090）上测量，采用Python标准time模块计时，取100帧的平均值。DeepSORT（18.3 FPS）相对ByteTrack（24.1 FPS）的速度劣势主要来自ReID特征提取的额外开销（每帧对每个检测目标执行一次ReID前向传播）。

## 补充5.8　RV1126B平台系统配置详述

**软件环境**

板端运行Buildroot Linux系统（内核版本4.19），预装Rockchip官方BSP（Board Support Package）和librknn_api.so推理库。C++推理程序依赖以下库：

- `librknn_api.so`：RKNN NPU推理接口，由Rockchip BSP提供；
- `libopencv_core.so`、`libopencv_imgproc.so`、`libopencv_imgcodecs.so`：图像读取和预处理（ARM交叉编译版OpenCV 4.5.5）；
- `libc`、`libstdc++`：C++标准库（由交叉编译工具链提供）。

所有依赖库均通过交叉编译静态链接或以共享库形式部署到板端 `/usr/lib/` 目录，推理程序以单可执行文件形式运行，不需要Python运行时。

**NPU性能特性**

RV1126B NPU的以下特性对部署方案设计有直接影响：

- 最大吞吐量：1TOPS（INT8）；
- 支持数据格式：INT8、FP16（FP16在NPU上执行，速度低于INT8约2倍）；
- 支持算子：Conv2D、DepthwiseConv、Pooling、Concat、Add、Sigmoid、Relu等主流算子；
- 不支持算子：部分动态形状算子、某些特殊激活函数（需回退至CPU执行）；
- 内存限制：NPU工作内存约128MB，超出部分会导致模型加载失败；
- IO带宽：ARM CPU与NPU间通过共享内存交换数据，吞吐量约为10GB/s，对于640×640×3的输入（约0.7MB），数据传输延迟可忽略。

YOLOv5s在RV1126B NPU上全部算子（包括Backbone和Neck的所有Conv和C3模块）均能在NPU上执行，无需回退至CPU，这是实现19.8ms均值延迟的重要保障。若存在不支持算子，部分计算需在ARM CPU上执行，会显著增加延迟（CPU与NPU之间的数据转移和CPU计算本身都会引入额外开销）。

**功耗测量**

在持续推理状态下（100%负载），通过外接电流表测量RV1126B的整机功耗约为2.8～3.2W（含CPU、NPU、内存、IO等所有组件），其中NPU活跃推理时的增量功耗约为1.2W。与GPU服务器（RTX 3090，推理功耗约80～120W）相比，功耗降低约25～40倍，体现了嵌入式NPU平台在功耗效率上的显著优势，适合长时间运行的边缘端应用场景。

## 补充3.8　数据增强对红外图像的适配分析

YOLOv5的默认数据增强策略主要面向RGB彩色图像设计，在应用于红外灰度图像时，部分增强策略的适配性需要评估。

**马赛克增强（Mosaic）**：在红外灰度图像上完全适用。红外图像的马赛克增强同样能提升小目标比例和样本多样性，且由于缺乏颜色信息，拼接边界处的颜色不连续问题（在RGB马赛克中有时会引入额外分布偏移）不复存在，可以说马赛克在灰度图上的增强更为"纯净"。

**HSV颜色空间增强（色调/饱和度/亮度抖动）**：在灰度图上，色调（H）和饱和度（S）通道的变换对单通道灰度图无实质影响（灰度图的H和S均为0），只有亮度（V）抖动能对图像产生有效扰动。YOLOv5在训练时进行了HSV颜色空间转换，对灰度图来说等价于单纯的亮度/对比度抖动，这一增强对红外图像是有效的——模拟不同照明强度下的亮度变化，提升模型对明暗对比度变化的鲁棒性。

**随机水平翻转（Horizontal Flip）**：对行人和车辆目标完全适用，红外场景中目标的左右对称性与可见光场景相同，翻转后标注坐标取镜像即可，不影响目标语义。

**随机透视变换（Perspective Transform）**：适用于模拟摄像头角度变化引起的透视畸变，对红外图像同样有效。透视变换主要改变目标的几何形状（宽高比轻微变化），与图像模态（彩色/灰度）无关。

**Cutout/CutMix等遮挡增强**：本文实验中未额外启用这些增强策略，仅使用YOLOv5默认增强配置。在存在大量目标遮挡的场景中，加入遮挡增强可能进一步提升模型对部分遮挡目标的召回率，但为保持消融实验的控变量要求，所有实验使用相同的默认增强策略。

总体而言，YOLOv5的默认数据增强策略在红外灰度图像场景下的适配性较好，无需针对性修改。这一发现简化了迁移工作，使研究重心可以集中在模型结构和损失函数的改进效果分析上。


## 补充第二章：相关工作深化——各改进模块的理论背景

### 补充2.9　Ghost模块与轻量化网络设计哲学

轻量化神经网络的设计哲学可以追溯到MobileNet系列引入的深度可分离卷积（Depthwise Separable Convolution）。深度可分离卷积将标准卷积分解为深度卷积（Depthwise Convolution，每个输入通道独立卷积）和逐点卷积（Pointwise Convolution，1×1卷积实现通道混合），计算量从 $H \times W \times C_{in} \times C_{out} \times k^2$ 降至 $H \times W \times C_{in} \times k^2 + H \times W \times C_{in} \times C_{out}$，节省比例约为 $1/C_{out} + 1/k^2$，对于 $C_{out}=64, k=3$ 的典型配置，节省约88%。

Ghost模块的创新在于换了一个视角：不是"如何更高效地做卷积"，而是"已有特征图中哪些是冗余的，可以从其他特征图廉价生成"。通过以标准卷积生成m个"本征特征图"，再经深度可分离卷积生成 $s \times m$ 个"幻象特征图"（s通常为2），总特征图数为 $m(1+s)$，而计算成本远低于直接生成 $m(1+s)$ 个特征图的标准卷积。Ghost-C3将此思想引入YOLOv5的C3模块，理论上可在近似精度的前提下降低35%以上的FLOPs。

然而，Ghost模块设计有一个隐性假设："大量特征图可以由少量本征特征图线性近似"。在信息量丰富的RGB图像中，这一假设在统计意义上成立（卷积特征图之间确实存在大量相关性）。但在灰度红外图像中，输入信息本身更为有限，特征图的相关性结构可能不同于RGB场景，Ghost近似的有效性假设被部分削弱，这是实验中Ghost-C3表现略逊于基线的理论解释之一。

### 补充2.10　注意力机制在目标检测中的作用机制

注意力机制（Attention Mechanism）在目标检测中的本质作用是：通过自适应地加权特征图的不同通道或不同空间位置，使网络将更多"计算资源"集中在对当前任务最有贡献的特征上。

在目标检测任务中，不同通道的特征图对应不同的视觉模式（边缘、纹理、形状等），注意力机制使网络能动态地强调与"当前目标类别最相关"的特征通道，抑制无关通道的干扰。这一机制在目标类别较少（本文仅2类）、目标形状规律性强的场景下的收益相对有限——因为网络本身就能在训练过程中学到哪些通道对行人/车辆更重要，注意力机制只是提供了一种更显式、更动态的机制来实现这一点。

在红外灰度图像场景下，注意力机制的空间维度价值可能比通道维度更为突出：红外图像中，目标与背景的对比度往往低于可见光图像，空间注意力（或含空间信息的通道注意力，如CoordAttention）有助于让网络主动关注"图像中可能存在目标的区域"，而非均匀处理所有空间位置的特征，这是Exp5（CoordAttention）略优于Exp4（SE注意力）的理论依据。

### 补充2.11　IoU系列损失函数的演化脉络与红外场景适用性分析

IoU损失函数的演化是近年来目标检测损失函数设计的重要研究线索。以下梳理从IoU到EIoU的演化逻辑：

**原始IoU损失**：$\mathcal{L}_{\text{IoU}} = 1 - \text{IoU}$，最直接地度量预测框与真实框的重叠质量，但对非重叠框梯度为零（无法更新参数，不适合训练初期框未重叠的情况）。

**GIoU**（2019）：$\mathcal{L}_{\text{GIoU}} = 1 - \text{IoU} + \frac{|C \setminus (A \cup B)|}{|C|}$，引入最小外接矩形C的非重叠面积惩罚，使非重叠框也有非零梯度，改善了收敛性，但当预测框完全包含在真实框内或真实框完全包含在预测框内时，惩罚项退化。

**DIoU**（2020）：在GIoU基础上用中心点距离替代非重叠面积惩罚，更直接地驱动预测框中心向真实框中心移动：$\mathcal{L}_{\text{DIoU}} = 1 - \text{IoU} + \frac{\rho^2(\mathbf{b}, \mathbf{b}^{gt})}{c^2}$，收敛速度更快。

**CIoU**（2020）：在DIoU基础上加入宽高比一致性约束：$\mathcal{L}_{\text{CIoU}} = \mathcal{L}_{\text{DIoU}} + \alpha v$，其中 $v = \frac{4}{\pi^2}\left(\arctan\frac{w^{gt}}{h^{gt}} - \arctan\frac{w}{h}\right)^2$，$\alpha = \frac{v}{(1-\text{IoU})+v}$。CIoU通过宽高比约束项使预测框的形状更快收敛至真实框形状，成为YOLOv5的默认损失函数。

**EIoU**（2022）：发现CIoU的宽高比约束 $v$ 在数学上同时包含宽度误差和高度误差（通过arctan函数耦合），当宽度误差较大时，梯度不一定正确地驱动宽度缩小（因为arctan的导数会同时影响w和h的更新）。EIoU将此约束分离为独立的宽度损失和高度损失，梯度分别更新宽度和高度，物理意义更明确，收敛更高效。

从演化脉络可见，EIoU是CIoU的自然延伸，在数学上更为完备，尤其适合目标宽高比变化规律的场景（行人高度方向波动、车辆宽度方向波动），理论上在红外灰度图像的行人/车辆检测场景具有优势，实验结果也验证了这一点。

## 补充第四章深化：跟踪算法在红外场景的适用性机理分析

### 补充4.8　DeepSORT外观特征在灰度红外场景的失效分析

DeepSORT引入外观特征的初衷是：当目标被遮挡或快速运动导致IoU关联失效时，通过外观相似度（ReID特征的余弦距离）重新建立轨迹与检测框的关联，减少ID切换。这一机制在可见光RGB场景下表现良好，因为行人的服装颜色、图案等在连续帧间保持高一致性，同一行人在不同帧的ReID特征余弦距离较小（通常<0.2），而不同行人的距离较大（通常>0.4），区分度足够。

然而，在灰度红外图像场景下，以下因素使ReID特征的区分度显著下降：

**色彩信息缺失**：服装颜色（最强的ReID视觉线索之一）在灰度图中完全丢失，不同行人如果穿着相似亮度的服装，其灰度图像的外观特征（ReID编码）会非常相似，导致余弦距离偏小，不同行人被错误关联。

**亮度动态变化**：近红外成像中，目标的亮度受到光源（路灯、车灯等）的动态影响，同一目标在不同帧的亮度可能变化较大，导致其ReID特征的帧间一致性降低，自身匹配的余弦距离增大，影响正确关联。

**目标分辨率限制**：当目标较小时（边界框面积<32×32像素），ReID特征提取器（通常在更大分辨率图像上训练）的特征提取质量下降，特征噪声增大，进一步削弱了区分度。

这些因素共同导致DeepSORT的ID Switch（54次）远多于ByteTrack（26次）——ReID特征不仅没有提供正确的跨帧关联，反而引入了额外的错误关联。对于红外灰度图像的跟踪任务，外观特征的设计方向可能需要转向基于形状/轮廓的特征（如目标边缘特征或骨架特征），而非基于颜色/纹理的传统ReID特征，这是未来可以深入研究的方向。

### 补充4.9　ByteTrack两轮匹配策略的数学描述

ByteTrack的两轮匹配算法可形式化描述如下：

**输入**：当前帧检测框集合 $D = D_{\text{high}} \cup D_{\text{low}}$，其中 $D_{\text{high}} = \{d \in D : \text{conf}(d) \geq \tau_{\text{high}}\}$，$D_{\text{low}} = \{d \in D : \tau_{\text{low}} \leq \text{conf}(d) < \tau_{\text{high}}\}$（$\tau_{\text{high}}=0.5$，$\tau_{\text{low}}=0.1$）；当前所有已确认轨迹集合 $T$。

**第一轮匹配**：
1. 计算代价矩阵 $C_1 \in \mathbb{R}^{|T| \times |D_{\text{high}}|}$，元素 $C_1[i,j] = 1 - \text{IoU}(T_i^{\text{pred}}, d_j)$；
2. 对 $C_1$ 应用匈牙利算法，得到最优匹配集合 $M_1 = \{(T_i, d_j)\}$，未匹配轨迹集合 $T^{\text{unmatched}} = T \setminus \{T_i : (T_i, d_j) \in M_1\}$，未匹配检测集合 $D_{\text{high}}^{\text{unmatched}}$；
3. 对 $M_1$ 中的轨迹，用对应检测框更新卡尔曼滤波状态；

**第二轮匹配**：
1. 计算代价矩阵 $C_2 \in \mathbb{R}^{|T^{\text{unmatched}}| \times |D_{\text{low}}|}$，元素同样为1-IoU；
2. 对 $C_2$ 应用匈牙利算法，成功匹配的轨迹-低置信度检测对 $M_2$ 中的轨迹恢复（避免因置信度暂时下降而终止轨迹）；

**轨迹更新**：未参与任何匹配的轨迹进入"缺失"计数递增，超过max_age则终止；未匹配的高置信度检测框 $D_{\text{high}}^{\text{unmatched}}$ 初始化为新轨迹（Tentative状态）。

ByteTrack的核心价值在于：当目标被遮挡时，检测器输出的置信度通常降低（因为遮挡使特征不完整），传统单轮匹配会丢弃这些低置信度框，导致轨迹中断并在遮挡结束后分配新ID（产生ID Switch）。ByteTrack的第二轮匹配将这些低置信度框"捡回来"用于轨迹续接，在不牺牲误检控制的前提下显著降低了ID Switch数量。

## 补充第五章深化：量化理论与工程实践的深度对接

### 补充5.9　INT8量化的数学基础

神经网络量化的核心是将浮点数 $x_f \in \mathbb{R}$ 映射为定点数 $x_q \in \mathbb{Z}_{[-128, 127]}$（INT8对称量化）：

$$x_q = \text{clamp}\left(\text{round}\left(\frac{x_f}{s}\right), -128, 127\right)$$

$$\tilde{x}_f = x_q \cdot s$$

其中 $s$ 为量化scale（步长），$\tilde{x}_f$ 为反量化近似值。量化误差为 $\epsilon = x_f - \tilde{x}_f$。

scale $s$ 的确定是量化精度的关键。不同校准算法本质上是不同的"如何最优地确定 $s$"的策略：

**Normal（MinMax）**：$s = \max(|x_f|) / 127$，直接以激活值绝对值的最大值确定scale。对于分布集中、无异常大值的激活分布，此法精度较好；但当存在少量极大异常值时，scale被拉大，导致大多数激活值只使用了量化范围的很小一部分（大量bit浪费在稀疏区域），精度损失显著。

**KL散度校准**：通过搜索最优截断阈值 $T \in [0, \max(|x_f|)]$，在截断后的分布 $p_T$ 与原始分布 $p$ 之间的KL散度 $\text{KL}(p \| q_T)$ 最小时，令 $s = T/127$（超出T的激活值被直接截断至127）。KL散度校准的物理意义是：在允许少量截断（舍弃异常大值的精确表示）的前提下，尽可能保留激活分布主体的信息量，适合分布存在尖峰或长尾的情况。

EIoU模型的置信度相关激活分布相比CIoU基线更为集中（峰值更高、尾部更薄），但Normal量化使用全局最大值确定scale，尾部的极小异常值（虽然概率密度极低）拉宽了scale，导致主体分布在量化后精度不足，出现置信度分数量化误差大于正常置信度范围（0.1～0.9）的问题。KL散度校准找到合适的截断阈值T后，绝大多数置信度分数都被精确量化，碎框问题消除。

### 补充5.10　C++推理框架的性能优化细节

在实现C++推理框架时，进行了以下性能优化，使全流程延迟满足20ms目标：

**预处理优化**：letterbox缩放使用OpenCV的 `cv::resize()` 函数（双线性插值），归一化直接通过 `img.convertTo(img_float, CV_32F, 1.0/255.0)` 批量处理，避免逐像素循环。整个预处理流程（包括内存分配和格式转换）控制在3.2ms均值。

**内存复用**：在100次连续推理中，预分配固定大小的输入/输出buffer，避免每帧重新malloc/free造成的内存碎片和系统调用开销。输入buffer大小为 $640 \times 640 \times 3 \times 4 = 4915200$ 字节（FP32格式，RKNN会在内部执行FP32→INT8转换）；三个输出buffer分别对应P3/P4/P5检测头的输出tensor，大小根据模型输出shape预分配。

**后处理优化**：sigmoid激活使用查表法（LUT，Lookup Table）替代逐元素计算 $\sigma(x) = 1/(1+e^{-x})$，在精度损失可忽略的前提下将sigmoid计算速度提升约3倍；NMS采用排序后的贪心算法（先将候选框按置信度降序排列，依次保留当前最高置信度框并过滤IoU超过阈值的框），对于典型场景下<200个候选框，此算法的时间复杂度 $O(n^2)$ 可接受，后处理总延迟约4.0ms。

**NPU推理预热**：在正式测速前，对同一模型执行10次推理（--warmup 10），使NPU缓存（指令缓存、权重缓存）进入稳定状态，避免冷启动引起的前几次推理延迟偏高（冷启动时NPU需要重新从DDR加载权重，额外耗时约5～10ms）。

