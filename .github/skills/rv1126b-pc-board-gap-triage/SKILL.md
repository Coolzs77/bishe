---
name: rv1126b-pc-board-gap-triage
description: 'RV1126B 板端与 PC 端视觉效果差异诊断技能。针对“板端不丝滑、框不贴合、误检多”场景，进行链路对齐、根因定位与修复闭环。关键词: rv1126b, pc-board 差异, 抖动, 误检, 框偏移, 量化.'
argument-hint: '示例: pc_video=outputs/tracking/.../result.mp4 board_video=outputs/out_video.mp4 board_image=outputs/out.png model=deploy/rv1126b_yolov5/model/best_eiou.rknn target_ms=20'
user-invocable: true
---

# RV1126B 板端/PC 视觉差异诊断技能

## 技能目标

用于定位并修复以下问题：

1. 板端视频观感不流畅、框抖动明显。
2. 板端图片/视频框与物体轮廓不贴合。
3. 板端误检/小碎框明显多于 PC 端。

## 关键前提（必须先统一）

1. 统一比较对象：PC 若是 **检测+跟踪**，板端也要启用跟踪，不能用“PC 跟踪视频 vs 板端纯检测视频”直接比。
2. 统一输入源：同一视频、同一分辨率、同一帧段。
3. 统一阈值：conf / nms 保持一致后再对比。
4. 统一模型：同一实验权重（如 exp07_eiou）与同一输入尺寸 640。

## 本项目已知高概率根因（按优先级）

### P0：比较链路不一致（最常见）

- PC `outputs/tracking/.../bytetrack.../result.mp4` 是“检测+ByteTrack+可见延迟”结果。
- 板端 `outputs/out_video.mp4` 来自 `deploy/rv1126b_yolov5/src/main_video.cc`，默认是**逐帧纯检测**。
- 结论：板端“抖动/不丝滑”在当前实现下是预期现象，不是单点 bug。

### P0：RKNN 输出格式与解码路径不一致

- INT8 场景必须优先 3-branch 输出；若误用单输出 `[1,25200,7]`，常见框质量劣化。
- 要求先看板端启动日志中 `n_output` 与输出 shape，再决定解码分支。

### P1：前后处理几何映射误差

- `convert_image_with_letterbox` 开启 `allow_slight_change` 对 resize 做对齐裁剪；
- 若缩放比与最终 resize 尺寸未严格一致，可能出现“框整体偏一点、不贴边”。

### P1：阈值与小框过滤策略不足

- 板端默认 `conf=0.35, nms=0.40`，且纯检测直接画框；
- 对远处密集目标会保留大量小框，视觉上更“乱”。

### P2：端到端链路瓶颈（与 NPU 推理无关）

- 板端逐帧 `cvtColor + malloc/free + 视频编码` 可能造成端到端卡顿；
- 需区分 NPU inference FPS 与端到端 FPS，避免误判为模型问题。

## 执行流程

### 阶段 0：建立对比契约（必须）

1. 固定视频片段（建议 >=300 帧）。
2. 记录 PC 与板端命令、阈值、模型路径。
3. 明确“硬比较对象”是纯检测还是检测+跟踪。

Completion check:
- 已形成可复现实验口径，不再混比。

### 阶段 1：板端模式识别

1. 检查板端日志：`n_output` 是否 3（推荐）或 1。
2. 若为 1 且为 INT8：优先判为高风险配置，建议回到 3-branch 导出与转换链路。
3. 打开 `RKNN_DEBUG_BOX=1` 采样前 20 帧，记录 decoded/NMS 数量与 letterbox 参数。

Completion check:
- 已明确是“模型输出问题”还是“后处理/阈值问题”。

### 阶段 2：几何与阈值排查

1. 核验 letterbox 映射参数（scale/x_pad/y_pad）是否合理。
2. 提高 conf（如 0.45~0.55）并适度提高 nms（如 0.45）做 A/B。
3. 增强小框过滤（最小宽高/面积）并复测误检率。

Completion check:
- 框贴合度与误检数量有定量改善。

### 阶段 3：时序稳定性排查

1. 若目标是“像 PC 一样丝滑”，板端必须引入跟踪或时序平滑。
2. 最低可行：IoU 关联 + TTL + 置信度 EMA（不改模型即可明显减抖）。
3. 区分并同时记录：
   - NPU 平均推理时延
   - 端到端 FPS（含编解码）

Completion check:
- 抖动显著下降，且性能指标可解释。

### 阶段 4：结论与修复决策

1. 给出“根因-证据-修复-收益”四联表。
2. 每轮仅改 1 个关键杠杆，保留前后数据。

Completion check:
- 可稳定复现并复盘“为什么变好”。

## 建议修复动作（本项目优先）

1. 保持 RKNN 3-branch 路径为主线，单输出仅作应急验证。
2. 在板端视频链路加入轻量时序稳定器（或移植 ByteTrack-lite）。
3. 将板端阈值调整为与 PC 对齐后再比较（先口径一致，再做调优）。
4. 对小框增加硬过滤，减少远处噪声框。
5. 速度分析时关闭视频写盘，先看纯推理瓶颈，再看 I/O。

## 关联资源

- [排障检查清单](./assets/triage-checklist.md)
- [诊断记录模板](./assets/diagnosis-template.md)
- [延迟解析脚本](../rv1126b-20ms-deploy-flow/scripts/parse_latency.py)

