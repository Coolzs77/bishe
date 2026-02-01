# RKNN量化模型说明

此目录用于存放转换后的RKNN量化模型文件。

## 文件说明

- `*.rknn` - RKNN格式的量化模型文件
- `calibration/` - 量化校准数据集

## 使用方法

1. 首先使用 `src/deploy/export_onnx.py` 导出ONNX模型
2. 然后使用 `src/deploy/convert_rknn.py` 转换为RKNN格式

详细使用说明请参考 `docs/部署指南.md`
