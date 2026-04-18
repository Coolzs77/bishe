---
name: external-rknn-rtsp-detect-reference
description: '外部参考技能：RTSP + RKNN 实时检测工作流（已按本项目结构标准化）。'
argument-hint: '示例: rtsp_url=rtsp://192.168.1.100/stream mode=reference'
user-invocable: true
---

# External RKNN RTSP Detect Reference

该技能是**外部技能的本地化参考版**，用于借鉴 RTSP 实时检测链路（启动/停止/状态/性能/故障排查）并迁移到本仓库的 RV1126B 工程实践。

## 来源

- 上游仓库：`johnponggit/learnEmbeddedAi`
- 原始文件：`src/app/demo_5_aiAgentDetect/SKILL.md`
- 原始链接：<https://github.com/johnponggit/learnEmbeddedAi/blob/8f8e240e92ce7cd232fa9d68024b7fba1a032deb/src/app/demo_5_aiAgentDetect/SKILL.md>

## 适用场景

1. 需要把离线视频推理扩展到 RTSP 实时流。
2. 需要补齐“检测服务化”环节（状态查询、日志、端口检查）。
3. 需要论文中“工程化部署能力”章节的可复现流程参考。

## 建议执行流程

1. 先对齐本仓库模型与后处理口径（输入尺寸、阈值、NMS）。
2. 再对齐流媒体 I/O（解码、检测、画框、编码、输出）时延分解。
3. 最后固化为脚本化入口（启动/停止/状态/日志）并记录指标。

## 输出要求

- 迁移后的命令清单（启动、停止、状态、性能）。
- 端到端 FPS 与 NPU 推理耗时分离报告。
- 常见故障路由（端口占用、模型初始化失败、输出异常）。

