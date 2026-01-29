#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
部署模块
提供model导出、convert和量化功能
"""

from .export_onnx import ONNXExporter, export_to_onnx, simplify_onnx_model, verify_onnx_model, test_onnx_inference, get_onnx_info
from .convert_rknn_obj import RKNNConverter, convert_to_rknn_obj, test_rknn_obj_model, create_calibration_dataset, get_rknn_obj_info
from .quantize import (
    QuantizationCalibrator, compute_quantization_params, 
    quantize_tensor, dequantize_tensor, evaluate_quantization_error, 
    ModelQuantizer, quantize_onnx_model
)

__all__ = [
    # ONNX导出
    'ONNXExporter',
    'export_to_onnx',
    'simplify_onnx_model',
    'verify_onnx_model',
    'test_onnx_inference',
    'get_onnx_info',
    # RKNNconvert
    'RKNNConverter',
    'convert_to_rknn_obj',
    'test_rknn_obj_model',
    'create_calibration_dataset',
    'get_rknn_obj_info',
    # 量化
    'QuantizationCalibrator',
    'compute_quantization_params',
    'quantize_tensor',
    'dequantize_tensor',
    'evaluate_quantization_error',
    'ModelQuantizer',
    'quantize_onnx_model',
]
