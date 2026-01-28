#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX模型导出模块

提供将PyTorch模型导出为ONNX格式的功能
"""

import os
from typing import List, Optional, Tuple, Dict, Any
import numpy as np


class ONNXExporter:
    """
    ONNX模型导出器
    
    将PyTorch模型导出为ONNX格式
    
    Attributes:
        model: PyTorch模型
        input_size: 输入尺寸 (height, width)
        opset_version: ONNX opset版本
        dynamic_axes: 动态轴配置
    """
    
    def __init__(
        self,
        input_size: Tuple[int, int] = (640, 640),
        opset_version: int = 12,
        dynamic_batch: bool = False
    ):
        """
        初始化ONNX导出器
        
        Args:
            input_size: 输入尺寸 (height, width)
            opset_version: ONNX opset版本
            dynamic_batch: 是否使用动态batch维度
        """
        self.input_size = input_size
        self.opset_version = opset_version
        self.dynamic_batch = dynamic_batch
        
        # 动态轴配置
        self.dynamic_axes = None
        if dynamic_batch:
            self.dynamic_axes = {
                'images': {0: 'batch'},
                'output': {0: 'batch'}
            }
    
    def export(
        self,
        model,
        output_path: str,
        input_names: List[str] = ['images'],
        output_names: List[str] = ['output'],
        simplify: bool = True
    ) -> str:
        """
        导出模型为ONNX格式
        
        Args:
            model: PyTorch模型
            output_path: 输出文件路径
            input_names: 输入节点名称列表
            output_names: 输出节点名称列表
            simplify: 是否简化ONNX模型
            
        Returns:
            导出的ONNX文件路径
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch未安装，请运行: pip install torch")
        
        # 确保模型在评估模式
        model.eval()
        
        # 创建虚拟输入
        dummy_input = torch.zeros(1, 3, self.input_size[0], self.input_size[1])
        
        # 获取设备
        device = next(model.parameters()).device
        dummy_input = dummy_input.to(device)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # 导出ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            opset_version=self.opset_version,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=self.dynamic_axes,
            do_constant_folding=True
        )
        
        print(f"模型已导出到: {output_path}")
        
        # 简化模型
        if simplify:
            output_path = simplify_onnx_model(output_path, output_path)
        
        return output_path


def export_to_onnx(
    model,
    output_path: str,
    input_size: Tuple[int, int] = (640, 640),
    opset_version: int = 12,
    dynamic_batch: bool = False,
    simplify: bool = True
) -> str:
    """
    导出PyTorch模型为ONNX格式
    
    Args:
        model: PyTorch模型
        output_path: 输出文件路径
        input_size: 输入尺寸 (height, width)
        opset_version: ONNX opset版本
        dynamic_batch: 是否使用动态batch维度
        simplify: 是否简化ONNX模型
        
    Returns:
        导出的ONNX文件路径
    """
    exporter = ONNXExporter(
        input_size=input_size,
        opset_version=opset_version,
        dynamic_batch=dynamic_batch
    )
    
    return exporter.export(model, output_path, simplify=simplify)


def simplify_onnx_model(input_path: str, output_path: Optional[str] = None) -> str:
    """
    简化ONNX模型
    
    使用onnx-simplifier简化模型，去除冗余操作
    
    Args:
        input_path: 输入ONNX文件路径
        output_path: 输出文件路径，如果为None则覆盖原文件
        
    Returns:
        简化后的ONNX文件路径
    """
    try:
        import onnx
        from onnxsim import simplify
    except ImportError:
        print("onnx-simplifier未安装，跳过简化步骤")
        return input_path
    
    if output_path is None:
        output_path = input_path
    
    # 加载模型
    model = onnx.load(input_path)
    
    # 简化模型
    model_simplified, check = simplify(model)
    
    if not check:
        print("警告: 简化后的模型验证失败，使用原始模型")
        return input_path
    
    # 保存简化后的模型
    onnx.save(model_simplified, output_path)
    print(f"简化后的模型已保存到: {output_path}")
    
    return output_path


def verify_onnx_model(model_path: str) -> bool:
    """
    验证ONNX模型
    
    检查ONNX模型的有效性
    
    Args:
        model_path: ONNX文件路径
        
    Returns:
        验证是否通过
    """
    try:
        import onnx
    except ImportError:
        print("onnx未安装，无法验证模型")
        return False
    
    try:
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        print(f"模型验证通过: {model_path}")
        return True
    except Exception as e:
        print(f"模型验证失败: {e}")
        return False


def test_onnx_inference(
    model_path: str,
    input_size: Tuple[int, int] = (640, 640),
    device: str = 'cpu'
) -> Tuple[bool, Optional[np.ndarray]]:
    """
    测试ONNX模型推理
    
    使用随机输入测试模型是否能正常运行
    
    Args:
        model_path: ONNX文件路径
        input_size: 输入尺寸 (height, width)
        device: 运行设备
        
    Returns:
        (是否成功, 输出结果)
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime未安装")
        return False, None
    
    try:
        # 选择执行提供者
        providers = ['CPUExecutionProvider']
        if device != 'cpu' and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        # 创建推理会话
        session = ort.InferenceSession(model_path, providers=providers)
        
        # 获取输入名称
        input_name = session.get_inputs()[0].name
        
        # 创建随机输入
        dummy_input = np.random.randn(1, 3, input_size[0], input_size[1]).astype(np.float32)
        
        # 运行推理
        outputs = session.run(None, {input_name: dummy_input})
        
        print(f"ONNX推理测试成功")
        print(f"输出形状: {[o.shape for o in outputs]}")
        
        return True, outputs[0]
        
    except Exception as e:
        print(f"ONNX推理测试失败: {e}")
        return False, None


def get_onnx_info(model_path: str) -> Dict[str, Any]:
    """
    获取ONNX模型信息
    
    Args:
        model_path: ONNX文件路径
        
    Returns:
        模型信息字典
    """
    try:
        import onnx
    except ImportError:
        return {'error': 'onnx未安装'}
    
    try:
        model = onnx.load(model_path)
        
        info = {
            'opset_version': model.opset_import[0].version,
            'inputs': [],
            'outputs': [],
            'nodes': len(model.graph.node),
            'file_size_mb': os.path.getsize(model_path) / (1024 * 1024)
        }
        
        # 输入信息
        for inp in model.graph.input:
            shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            info['inputs'].append({
                'name': inp.name,
                'shape': shape
            })
        
        # 输出信息
        for out in model.graph.output:
            shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
            info['outputs'].append({
                'name': out.name,
                'shape': shape
            })
        
        return info
        
    except Exception as e:
        return {'error': str(e)}
