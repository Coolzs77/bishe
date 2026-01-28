#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX模型导出模块
将PyTorch模型导出为ONNX格式
"""

import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import sys


def 导出ONNX(
    模型路径: str,
    输出路径: str,
    输入尺寸: int = 640,
    批量大小: int = 1,
    动态批量: bool = False,
    opset版本: int = 12,
    简化: bool = True
) -> bool:
    """
    将PyTorch模型导出为ONNX格式
    
    参数:
        模型路径: PyTorch模型路径 (.pt)
        输出路径: ONNX输出路径 (.onnx)
        输入尺寸: 输入图像尺寸
        批量大小: 批量大小
        动态批量: 是否支持动态批量
        opset版本: ONNX opset版本
        简化: 是否简化ONNX图
    
    返回:
        是否成功
    """
    try:
        import torch
    except ImportError:
        print("错误: 请先安装PyTorch")
        return False
    
    print(f"导出ONNX模型...")
    print(f"  输入: {模型路径}")
    print(f"  输出: {输出路径}")
    print(f"  输入尺寸: {输入尺寸}x{输入尺寸}")
    print(f"  批量大小: {批量大小}")
    print(f"  opset版本: {opset版本}")
    
    try:
        # 加载模型
        print("\n加载PyTorch模型...")
        模型 = torch.load(模型路径, map_location='cpu')
        
        if isinstance(模型, dict):
            if 'model' in 模型:
                模型 = 模型['model']
            elif 'state_dict' in 模型:
                print("警告: 模型文件只包含state_dict，需要模型定义")
                return False
        
        模型.eval()
        
        # 创建示例输入
        示例输入 = torch.randn(批量大小, 3, 输入尺寸, 输入尺寸)
        
        # 配置动态轴
        动态轴 = None
        if 动态批量:
            动态轴 = {
                'images': {0: 'batch'},
                'output': {0: 'batch'},
            }
        
        # 确保输出目录存在
        Path(输出路径).parent.mkdir(parents=True, exist_ok=True)
        
        # 导出ONNX
        print("\n导出ONNX...")
        torch.onnx.export(
            模型,
            示例输入,
            输出路径,
            verbose=False,
            opset_version=opset版本,
            do_constant_folding=True,
            input_names=['images'],
            output_names=['output'],
            dynamic_axes=动态轴,
        )
        
        print(f"ONNX模型已保存: {输出路径}")
        
        # 简化ONNX
        if 简化:
            简化ONNX模型(输出路径)
        
        # 验证ONNX
        验证ONNX模型(输出路径)
        
        return True
        
    except Exception as e:
        print(f"导出失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def 简化ONNX模型(模型路径: str) -> bool:
    """
    简化ONNX模型
    
    参数:
        模型路径: ONNX模型路径
    
    返回:
        是否成功
    """
    try:
        import onnx
        from onnxsim import simplify
        
        print("\n简化ONNX模型...")
        
        模型 = onnx.load(模型路径)
        简化模型, 检查结果 = simplify(模型)
        
        if 检查结果:
            onnx.save(简化模型, 模型路径)
            print("  简化成功")
            return True
        else:
            print("  简化验证失败，保留原模型")
            return False
            
    except ImportError:
        print("  跳过简化: onnx-simplifier未安装")
        return False
    except Exception as e:
        print(f"  简化失败: {e}")
        return False


def 验证ONNX模型(模型路径: str) -> bool:
    """
    验证ONNX模型
    
    参数:
        模型路径: ONNX模型路径
    
    返回:
        是否有效
    """
    try:
        import onnx
        
        print("\n验证ONNX模型...")
        
        模型 = onnx.load(模型路径)
        onnx.checker.check_model(模型)
        
        # 打印模型信息
        print("  模型验证通过")
        print(f"  输入: {[inp.name for inp in 模型.graph.input]}")
        print(f"  输出: {[out.name for out in 模型.graph.output]}")
        
        # 计算模型大小
        文件大小 = Path(模型路径).stat().st_size / (1024 * 1024)
        print(f"  文件大小: {文件大小:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"  验证失败: {e}")
        return False


def 测试ONNX推理(
    模型路径: str,
    输入尺寸: int = 640,
    迭代次数: int = 10
) -> dict:
    """
    测试ONNX模型推理
    
    参数:
        模型路径: ONNX模型路径
        输入尺寸: 输入图像尺寸
        迭代次数: 测试迭代次数
    
    返回:
        测试结果字典
    """
    try:
        import onnxruntime as ort
        import time
        
        print(f"\n测试ONNX推理性能 ({迭代次数}次)...")
        
        # 创建推理会话
        会话 = ort.InferenceSession(
            模型路径,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        输入名 = 会话.get_inputs()[0].name
        
        # 创建测试输入
        测试输入 = np.random.randn(1, 3, 输入尺寸, 输入尺寸).astype(np.float32)
        
        # 预热
        for _ in range(3):
            会话.run(None, {输入名: 测试输入})
        
        # 计时
        时间列表 = []
        for _ in range(迭代次数):
            开始 = time.time()
            会话.run(None, {输入名: 测试输入})
            结束 = time.time()
            时间列表.append((结束 - 开始) * 1000)
        
        平均时间 = np.mean(时间列表)
        最小时间 = np.min(时间列表)
        最大时间 = np.max(时间列表)
        
        print(f"  平均推理时间: {平均时间:.2f} ms")
        print(f"  最小推理时间: {最小时间:.2f} ms")
        print(f"  最大推理时间: {最大时间:.2f} ms")
        print(f"  FPS: {1000/平均时间:.1f}")
        
        return {
            'avg_ms': 平均时间,
            'min_ms': 最小时间,
            'max_ms': 最大时间,
            'fps': 1000 / 平均时间,
        }
        
    except Exception as e:
        print(f"  推理测试失败: {e}")
        return None


class ONNX导出器:
    """
    ONNX模型导出器类
    """
    
    def __init__(
        self,
        输入尺寸: int = 640,
        opset版本: int = 12,
        动态批量: bool = False
    ):
        """
        初始化导出器
        
        参数:
            输入尺寸: 输入图像尺寸
            opset版本: ONNX opset版本
            动态批量: 是否支持动态批量
        """
        self.输入尺寸 = 输入尺寸
        self.opset版本 = opset版本
        self.动态批量 = 动态批量
    
    def 导出(self, 模型路径: str, 输出路径: str = None) -> str:
        """
        导出模型
        
        参数:
            模型路径: PyTorch模型路径
            输出路径: ONNX输出路径（可选）
        
        返回:
            输出文件路径
        """
        if 输出路径 is None:
            输出路径 = str(Path(模型路径).with_suffix('.onnx'))
        
        成功 = 导出ONNX(
            模型路径=模型路径,
            输出路径=输出路径,
            输入尺寸=self.输入尺寸,
            动态批量=self.动态批量,
            opset版本=self.opset版本,
        )
        
        if 成功:
            return 输出路径
        return None
    
    def 测试(self, 模型路径: str) -> dict:
        """
        测试导出的模型
        
        参数:
            模型路径: ONNX模型路径
        
        返回:
            测试结果
        """
        return 测试ONNX推理(模型路径, self.输入尺寸)
