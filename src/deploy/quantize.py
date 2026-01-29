#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model量化模块

提供INT8量化功能，用于model压缩和加速
"""

import os
from typing import List, Optional, Tuple, Dict, Any, Callable
import numpy as np


class QuantizationCalibrator:
    """
    量化校准器
    
    收集校准data并计算量化参数
    
    Attributes:
        num_samples: 校准样本count
        percentile: 百分位数（用于确定量化范围）
        data_collector: data收集列表
    """
    
    def __init__(self, num_samples: int = 100, percentile: float = 99.99):
        """
        初始化量化校准器
        
        Args:
            num_samples: 校准样本count
            percentile: 百分位数，用于计算量化范围
        """
        self.num_samples = num_samples
        self.percentile = percentile
        self.data_collector: List[np.ndarray] = []
    
    def collect(self, data: np.ndarray) -> None:
        """
        收集校准data
        
        Args:
            data: inputdata
        """
        if len(self.data_collector) < self.num_samples:
            self.data_collector.append(data.copy())
    
    def compute_scale_zero_point(self, symmetric: bool = True) -> Tuple[float, int]:
        """
        计算量化参数（scale和zero_point）
        
        Args:
            symmetric: 是否使用对称量化
            
        Returns:
            (scale, zero_point)
        """
        if len(self.data_collector) == 0:
            return 1.0, 0
        
        # 合并所有收集的data
        all_data = np.concatenate([d.flatten() for d in self.data_collector])
        
        # 计算范围
        if symmetric:
            # 对称量化
            abs_max = np.percentile(np.abs(all_data), self.percentile)
            scale = abs_max / 127.0
            zero_point = 0
        else:
            # 非对称量化
            min_val = np.percentile(all_data, 100 - self.percentile)
            max_val = np.percentile(all_data, self.percentile)
            scale = (max_val - min_val) / 255.0
            zero_point = int(round(-min_val / scale))
            zero_point = np.clip(zero_point, 0, 255)
        
        return float(scale), int(zero_point)
    
    def reset(self) -> None:
        """重置校准器"""
        self.data_collector = []


def compute_quantization_params(
    data: np.ndarray,
    bits: int = 8,
    symmetric: bool = True,
    percentile: float = 99.99
) -> Tuple[float, int]:
    """
    计算量化参数
    
    Args:
        data: inputdata
        bits: 量化位数
        symmetric: 是否使用对称量化
        percentile: 百分位数
        
    Returns:
        (scale, zero_point)
    """
    data = data.flatten()
    
    if symmetric:
        # 对称量化
        abs_max = np.percentile(np.abs(data), percentile)
        qmax = (2 ** (bits - 1)) - 1
        scale = abs_max / qmax if abs_max > 0 else 1.0
        zero_point = 0
    else:
        # 非对称量化
        min_val = np.percentile(data, 100 - percentile)
        max_val = np.percentile(data, percentile)
        qmax = (2 ** bits) - 1
        scale = (max_val - min_val) / qmax if (max_val - min_val) > 0 else 1.0
        zero_point = int(round(-min_val / scale))
        zero_point = np.clip(zero_point, 0, qmax)
    
    return float(scale), int(zero_point)


def quantize_tensor(
    tensor: np.ndarray,
    scale: float,
    zero_point: int = 0,
    bits: int = 8,
    symmetric: bool = True
) -> np.ndarray:
    """
    量化张量
    
    Args:
        tensor: input张量
        scale: 缩放因子
        zero_point: 零点
        bits: 量化位数
        symmetric: 是否使用对称量化
        
    Returns:
        量化后的张量
    """
    if symmetric:
        qmax = (2 ** (bits - 1)) - 1
        qmin = -qmax - 1
    else:
        qmax = (2 ** bits) - 1
        qmin = 0
    
    # 量化
    quantized = np.round(tensor / scale + zero_point)
    quantized = np.clip(quantized, qmin, qmax).astype(np.int8 if symmetric else np.uint8)
    
    return quantized


def dequantize_tensor(
    tensor: np.ndarray,
    scale: float,
    zero_point: int = 0
) -> np.ndarray:
    """
    反量化张量
    
    Args:
        tensor: 量化后的张量
        scale: 缩放因子
        zero_point: 零点
        
    Returns:
        反量化后的张量
    """
    return (tensor.astype(np.float32) - zero_point) * scale


def evaluate_quantization_error(
    original: np.ndarray,
    quantized: np.ndarray,
    scale: float,
    zero_point: int = 0
) -> Dict[str, float]:
    """
    evaluate量化误差
    
    Args:
        original: 原始张量
        quantized: 量化后的张量
        scale: 缩放因子
        zero_point: 零点
        
    Returns:
        误差metrics字典
    """
    # 反量化
    reconstructed = dequantize_tensor(quantized, scale, zero_point)
    
    # 计算误差
    diff = original - reconstructed
    
    metrics = {
        'mse': float(np.mean(diff ** 2)),
        'mae': float(np.mean(np.abs(diff))),
        'max_error': float(np.max(np.abs(diff))),
        'snr': float(10 * np.log10(np.mean(original ** 2) / (np.mean(diff ** 2) + 1e-10)))
    }
    
    return metrics


class ModelQuantizer:
    """
    model量化器
    
    提供model量化功能
    
    Attributes:
        calibrator: 量化校准器
        bits: 量化位数
        symmetric: 是否使用对称量化
        layer_scales: 各层的量化参数
    """
    
    def __init__(
        self,
        bits: int = 8,
        symmetric: bool = True,
        percentile: float = 99.99
    ):
        """
        初始化model量化器
        
        Args:
            bits: 量化位数
            symmetric: 是否使用对称量化
            percentile: 百分位数
        """
        self.bits = bits
        self.symmetric = symmetric
        self.percentile = percentile
        self.layer_scales: Dict[str, Tuple[float, int]] = {}
        self.calibrators: Dict[str, QuantizationCalibrator] = {}
    
    def add_calibration_hook(self, layer_name: str) -> Callable:
        """
        添加校准钩子
        
        Args:
            layer_name: 层name
            
        Returns:
            校准钩子函数
        """
        if layer_name not in self.calibrators:
            self.calibrators[layer_name] = QuantizationCalibrator(percentile=self.percentile)
        
        calibrator = self.calibrators[layer_name]
        
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            calibrator.collect(output.detach().cpu().numpy())
        
        return hook
    
    def calibrate(self) -> None:
        """
        执行校准，计算各层的量化参数
        """
        for layer_name, calibrator in self.calibrators.items():
            scale, zero_point = calibrator.compute_scale_zero_point(self.symmetric)
            self.layer_scales[layer_name] = (scale, zero_point)
            print(f"层 {layer_name}: scale={scale:.6f}, zero_point={zero_point}")
    
    def quantize_weights(self, weights: np.ndarray, layer_name: str) -> Tuple[np.ndarray, float, int]:
        """
        量化权重
        
        Args:
            weights: 权重张量
            layer_name: 层name
            
        Returns:
            (量化后的权重, scale, zero_point)
        """
        # 计算权重的量化参数（每通道或全局）
        scale, zero_point = compute_quantization_params(
            weights, self.bits, self.symmetric, self.percentile
        )
        
        # 量化
        quantized = quantize_tensor(weights, scale, zero_point, self.bits, self.symmetric)
        
        return quantized, scale, zero_point
    
    def save_quantization_params(self, filepath: str) -> None:
        """
        保存量化参数
        
        Args:
            filepath: 保存路径
        """
        import json
        
        params = {
            'bits': self.bits,
            'symmetric': self.symmetric,
            'percentile': self.percentile,
            'layer_scales': {k: {'scale': v[0], 'zero_point': v[1]} for k, v in self.layer_scales.items()}
        }
        
        with open(filepath, 'w') as f:
            json.dump(params, f, indent=2)
        
        print(f"量化参数已保存到: {filepath}")
    
    def load_quantization_params(self, filepath: str) -> None:
        """
        加载量化参数
        
        Args:
            filepath: 参数文件路径
        """
        import json
        
        with open(filepath, 'r') as f:
            params = json.load(f)
        
        self.bits = params.get('bits', 8)
        self.symmetric = params.get('symmetric', True)
        self.percentile = params.get('percentile', 99.99)
        
        for layer_name, scales in params.get('layer_scales', {}).items():
            self.layer_scales[layer_name] = (scales['scale'], scales['zero_point'])
        
        print(f"量化参数已加载: {filepath}")


def quantize_onnx_model(
    input_path: str,
    output_path: str,
    calibration_data: Optional[List[np.ndarray]] = None,
    quantization_type: str = 'int8'
) -> str:
    """
    量化ONNXmodel
    
    使用ONNX Runtime的量化工具
    
    Args:
        input_path: inputONNXmodel路径
        output_path: output量化model路径
        calibration_data: 校准data列表
        quantization_type: 量化类型 ('int8', 'uint8')
        
    Returns:
        量化后的model路径
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, quantize_static, CalibrationDataReader
    except ImportError:
        print("onnxruntime-quantization未安装")
        return input_path
    
    if calibration_data is None or len(calibration_data) == 0:
        # 动态量化
        quantize_dynamic(
            input_path,
            output_path,
            weight_type=np.int8 if quantization_type == 'int8' else np.uint8
        )
    else:
        # 静态量化
        class DataReader(CalibrationDataReader):
            def __init__(self, data_list):
                self.data = data_list
                self.index = 0
            
            def get_next(self):
                if self.index < len(self.data):
                    data = {'images': self.data[self.index]}
                    self.index += 1
                    return data
                return None
        
        reader = DataReader(calibration_data)
        quantize_static(
            input_path,
            output_path,
            reader
        )
    
    print(f"量化model已保存到: {output_path}")
    return output_path
