#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
部署模块
提供模型导出、转换和量化功能
"""

# 导入 ONNX 相关功能
from .export_onnx import (
    ONNXExporter,
    export_to_onnx,
    simplify_onnx_model,
    verify_onnx_model,
    test_onnx_inference,
    get_onnx_info
)

# 修复：修改导入源为 .convert_rknn，并使用正确的函数名
# 原来的 .convert_rknn_obj 是错误的文件名
from .convert_rknn import (
    RKNNConverter,
    convert_to_rknn,  # 修正：原来是 convert_to_rknn_obj
    test_rknn_model,  # 修正：原来是 test_rknn_obj_model
    create_calibration_dataset,
    get_rknn_info  # 修正：原来是 get_rknn_obj_info
)

# 导入量化相关功能
from .quantize import (
    QuantizationCalibrator,
    compute_quantization_params,
    quantize_tensor,
    dequantize_tensor,
    evaluate_quantization_error,
    ModelQuantizer,
    quantize_onnx_model
)

__all__ = [
    # ONNX导出
    'ONNXExporter',
    'export_to_onnx',
    'simplify_onnx_model',
    'verify_onnx_model',
    'test_onnx_inference',
    'get_onnx_info',

    # RKNN转换 (已修正名称)
    'RKNNConverter',
    'convert_to_rknn',
    'test_rknn_model',
    'create_calibration_dataset',
    'get_rknn_info',

    # 量化
    'QuantizationCalibrator',
    'compute_quantization_params',
    'quantize_tensor',
    'dequantize_tensor',
    'evaluate_quantization_error',
    'ModelQuantizer',
    'quantize_onnx_model',
]