---
name: rv1126b-20ms-deploy-flow
description: 'RV1126B YOLOv5 部署与性能优化工作流，目标 NPU 平均推理 <=20ms。Use for ONNX to RKNN conversion, cross-compilation, board profiling, bottleneck triage, controlled iteration. 关键词: rv1126b, rknn, 部署, 量化, 延迟, 20ms.'
argument-hint: '示例: model=deploy/rv1126b_yolov5/model/best_eiou.onnx baseline_ms=27.6 map50_drop_budget=1.0 fixed_imgsz=640'
user-invocable: true
---

# RV1126B <=20ms 部署技能流（中文注释版）

## 技能产出

1. 一套可复现的 RV1126B 部署包（可直接上板运行）。
2. 一份延迟报告（mean、p50、p90、p95）。
3. 一份优化决策日志（每次改动、原因、收益）。
4. 一份最终验收结论（是否达到 <=20.0 ms 硬指标）。

## 何时使用

- 你在做 YOLOv5 红外检测到 RV1126B 的部署。
- 板端延迟超过目标，需要结构化优化闭环。
- 你需要可复盘、可写进论文/里程碑的结果。

## 输入要求

- ONNX 模型路径（优先 3-branch 输出格式）。
- 当前基线延迟（如有）。
- 精度预算（默认 mAP50 绝对下降 <= 1.0）。
- 输入尺寸固定 640（不允许 608/576 分支）。
- 目标硬指标：NPU 平均推理 <= 20.0 ms。

## 项目默认基线约定

- 主力部署模型: `ablation_exp07_eiou`。
- 轻量候选模型: `ablation_exp09_ghost_eiou`。
- 严格控制变量：每轮只改一个关键杠杆。

## 执行流程

### 阶段 0：先锁定测速口径（Measurement Contract）

1. 以板端日志作为 NPU 推理延迟真值来源。
2. 固定测试视频（建议 >=300 帧）。
3. 对比速度时固定 conf 与 NMS 阈值。
4. 明确验收硬指标只看 NPU 平均推理。
5. 端到端视频链路只记录观察，不作硬门槛。
6. 使用 [报告模板](./assets/report-template.md) 记录元数据。

Completion check:
- 优化前，测速设置与硬/软指标边界都已固定。

### 阶段 1：先验模型导出格式（再谈优化）

1. 按项目部署流程导出 ONNX。
2. 确认输出为 3-branch 原始头（P3/P4/P5）。
3. INT8 部署时禁用单输出 `[1,25200,7]`，避免置信度量化塌陷。

Completion check:
- ONNX 输出格式通过校验，且与当前 C++ 后处理兼容。

### 阶段 2：RKNN 转换与量化

1. 使用 INT8 和有代表性的校准集把 ONNX 转为 RKNN。
2. 校准集要覆盖实际部署场景分布。
3. 若出现 0 检测，优先排查输出格式和校准覆盖度。

Completion check:
- RKNN 在已知样本上可稳定输出非空检测结果。

### 阶段 3：交叉编译与运行包组装

1. 使用 `deploy/rv1126b_yolov5/build_rv1126b.sh` 编译。
2. 确保使用与板端匹配的 RGA 运行库（避免不兼容 `librga.so`）。
3. 组装运行包（`bin`、`lib`、`model`、标签、测试媒体）。

Completion check:
- 运行包可上板运行，且无缺库错误。

### 阶段 4：板端基线测速

1. 在板端跑固定测试媒体并采集日志。
2. 用 [延迟解析脚本](./scripts/parse_latency.py) 提取指标。
3. 在 [报告模板](./assets/report-template.md) 填写 baseline。

Example parser command:

```bash
python .github/skills/rv1126b-20ms-deploy-flow/scripts/parse_latency.py /path/to/board_run.log --target-ms 20
```

Completion check:
- 你已拿到 mean/p50/p90/p95，可进入优化决策。

### 阶段 5：瓶颈分诊（核心分支）

参考 [优化矩阵](./references/optimization-matrix.md)，每次只动一个杠杆。

1. 若 `avg_infer_ms <= 20.0`，进入阶段 6。
2. 若 `avg_infer_ms > 20.0`，按优先级依次排查：
   - 运行路径正确性：RGA/NPU 路径与库兼容性。
   - 量化质量：校准集多样性、输出数值是否正常。
   - 模型杠杆：切换 `ablation_exp09_ghost_eiou` 候选。
   - C++ 管线优化：减拷贝、预分配、裁剪非必要路径。
3. 每次改动后，严格复用同一测速口径重测。

Completion check:
- 每次提速都能解释“为什么快了”，并有前后数据。

### 阶段 6：精度与稳定性门禁

1. 用固定验证子集对比，防止过优化。
2. `person` 与 `car` 不允许灾难性下降。
3. 板端至少重复 3 次测速，确认波动可接受。

Recommended acceptance:
- 硬指标：NPU 平均推理 <= 20.0 ms。
- 硬指标：mAP50 绝对下降 <= 1.0。
- 软观察：p90 与端到端 FPS 不应出现异常退化。

Completion check:
- 速度目标与精度预算同时满足。

### 阶段 7：最终交付物

1. 最终 `.rknn` 与部署运行包。
2. 填写完整的 [报告模板](./assets/report-template.md)。
3. 优化决策总结（有效/无效/保留方案）。

Completion check:
- 其他工程师可按你的命令和报告复现实验结果。

## 快速故障路由

- 现象：`0 detections`
  - 先查 ONNX 输出格式与量化校准覆盖。
- 现象：`rga2 get info failed`
  - 更换板端匹配 `librga.so` 后重新编译。
- 现象：`rknn_init failed`
  - 核对目标平台与运行时库版本匹配关系。
- 现象：端到端 FPS 低，但 inference ms 正常
  - 单独剖析解码/编码/I/O，不要误判为 NPU 瓶颈。

## 关联资源

- [优化矩阵](./references/optimization-matrix.md)
- [部署报告模板](./assets/report-template.md)
- [延迟解析脚本](./scripts/parse_latency.py)
