# 章节写作提示模板（可直接喂给 Copilot）

## 模板1：检测实验章节
请根据以下证据起草“检测实验与结果分析”章节，要求学术风格、包含小节标题、图表占位：
- 证据路径：`outputs/detection/*`, `scripts/evaluate/eval_detection.py`, `configs/eval_detection.yaml`
- 需要覆盖：实验设置、对比指标、消融结论、误差分析
- 字数：3000~4000

## 模板2：跟踪章节
请根据以下证据起草“跟踪方法与实验分析”章节，要求解释 tracker 选择依据与稳定性分析：
- 证据路径：`scripts/evaluate/eval_tracking.py`, `src/tracking/*`, `outputs/tracking/*`
- 需要覆盖：算法流程、参数策略、match_rate/FPS/ID switch proxy 解读
- 字数：2500~3500

## 模板3：部署章节
请根据以下证据起草“RV1126B 部署方案设计与实现”章节，要求流程完整、可复现：
- 证据路径：`deploy/rv1126b_yolov5/*`, `.github/skills/rv1126b-20ms-deploy-flow/*`
- 需要覆盖：ONNX 导出、RKNN 量化、交叉编译、板端测速与优化迭代
- 字数：3000~4500

