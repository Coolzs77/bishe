#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
目标检测器基类
定义检测器的通用接口
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
from dataclasses import dataclass


@dataclass
class 检测结果:
    """
    检测结果数据类
    """
    边界框: np.ndarray      # [N, 4] (x1, y1, x2, y2)
    置信度: np.ndarray      # [N]
    类别: np.ndarray        # [N]
    类别名称: List[str]     # 类别名称列表
    
    def __len__(self) -> int:
        return len(self.边界框)
    
    def __getitem__(self, idx: int) -> Dict:
        return {
            '边界框': self.边界框[idx],
            '置信度': float(self.置信度[idx]),
            '类别': int(self.类别[idx]),
            '类别名称': self.类别名称[int(self.类别[idx])] if self.类别名称 else None,
        }
    
    def 过滤(self, 置信度阈值: float) -> '检测结果':
        """按置信度阈值过滤结果"""
        掩码 = self.置信度 >= 置信度阈值
        return 检测结果(
            边界框=self.边界框[掩码],
            置信度=self.置信度[掩码],
            类别=self.类别[掩码],
            类别名称=self.类别名称,
        )
    
    def 按类别过滤(self, 类别ID: int) -> '检测结果':
        """按类别ID过滤结果"""
        掩码 = self.类别 == 类别ID
        return 检测结果(
            边界框=self.边界框[掩码],
            置信度=self.置信度[掩码],
            类别=self.类别[掩码],
            类别名称=self.类别名称,
        )
    
    def 转换为列表(self) -> List[Dict]:
        """转换为字典列表"""
        结果列表 = []
        for i in range(len(self)):
            结果列表.append(self[i])
        return 结果列表


class 检测器基类(ABC):
    """
    目标检测器的抽象基类
    """
    
    def __init__(
        self,
        模型路径: str,
        类别列表: List[str],
        置信度阈值: float = 0.5,
        NMS阈值: float = 0.45,
        设备: str = 'cuda'
    ):
        """
        初始化检测器
        
        参数:
            模型路径: 模型权重文件路径
            类别列表: 检测类别名称列表
            置信度阈值: 置信度过滤阈值
            NMS阈值: 非极大值抑制阈值
            设备: 运行设备 ('cuda' 或 'cpu')
        """
        self.模型路径 = 模型路径
        self.类别列表 = 类别列表
        self.类别数 = len(类别列表)
        self.置信度阈值 = 置信度阈值
        self.NMS阈值 = NMS阈值
        self.设备 = 设备
        self.模型 = None
    
    @abstractmethod
    def 加载模型(self):
        """加载模型权重"""
        pass
    
    @abstractmethod
    def 预处理(self, 图像: np.ndarray) -> np.ndarray:
        """
        图像预处理
        
        参数:
            图像: BGR格式的输入图像
        
        返回:
            预处理后的图像张量
        """
        pass
    
    @abstractmethod
    def 推理(self, 输入: np.ndarray) -> np.ndarray:
        """
        执行模型推理
        
        参数:
            输入: 预处理后的输入张量
        
        返回:
            模型原始输出
        """
        pass
    
    @abstractmethod
    def 后处理(self, 输出: np.ndarray, 原始尺寸: Tuple[int, int]) -> 检测结果:
        """
        后处理模型输出
        
        参数:
            输出: 模型原始输出
            原始尺寸: 原始图像尺寸 (高, 宽)
        
        返回:
            检测结果对象
        """
        pass
    
    def 检测(self, 图像: np.ndarray) -> 检测结果:
        """
        执行完整的检测流程
        
        参数:
            图像: BGR格式的输入图像
        
        返回:
            检测结果对象
        """
        原始尺寸 = 图像.shape[:2]
        
        # 预处理
        输入 = self.预处理(图像)
        
        # 推理
        输出 = self.推理(输入)
        
        # 后处理
        结果 = self.后处理(输出, 原始尺寸)
        
        return 结果
    
    def 批量检测(self, 图像列表: List[np.ndarray]) -> List[检测结果]:
        """
        批量检测
        
        参数:
            图像列表: 图像列表
        
        返回:
            检测结果列表
        """
        return [self.检测(图像) for 图像 in 图像列表]
    
    def 设置阈值(self, 置信度阈值: float = None, NMS阈值: float = None):
        """
        设置检测阈值
        
        参数:
            置信度阈值: 置信度过滤阈值
            NMS阈值: NMS阈值
        """
        if 置信度阈值 is not None:
            self.置信度阈值 = 置信度阈值
        if NMS阈值 is not None:
            self.NMS阈值 = NMS阈值
    
    def 获取类别名称(self, 类别ID: int) -> str:
        """获取类别名称"""
        if 0 <= 类别ID < len(self.类别列表):
            return self.类别列表[类别ID]
        return f"unknown_{类别ID}"
    
    @staticmethod
    def NMS(
        边界框: np.ndarray, 
        置信度: np.ndarray, 
        IoU阈值: float = 0.45
    ) -> np.ndarray:
        """
        非极大值抑制
        
        参数:
            边界框: [N, 4] 边界框数组
            置信度: [N] 置信度数组
            IoU阈值: IoU阈值
        
        返回:
            保留的索引数组
        """
        if len(边界框) == 0:
            return np.array([], dtype=np.int64)
        
        x1 = 边界框[:, 0]
        y1 = 边界框[:, 1]
        x2 = 边界框[:, 2]
        y2 = 边界框[:, 3]
        
        面积 = (x2 - x1) * (y2 - y1)
        排序索引 = np.argsort(-置信度)
        
        保留列表 = []
        
        while len(排序索引) > 0:
            当前索引 = 排序索引[0]
            保留列表.append(当前索引)
            
            if len(排序索引) == 1:
                break
            
            其他索引 = 排序索引[1:]
            
            # 计算IoU
            xx1 = np.maximum(x1[当前索引], x1[其他索引])
            yy1 = np.maximum(y1[当前索引], y1[其他索引])
            xx2 = np.minimum(x2[当前索引], x2[其他索引])
            yy2 = np.minimum(y2[当前索引], y2[其他索引])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            交集 = w * h
            
            IoU = 交集 / (面积[当前索引] + 面积[其他索引] - 交集 + 1e-10)
            
            # 保留IoU小于阈值的框
            排序索引 = 其他索引[IoU <= IoU阈值]
        
        return np.array(保留列表, dtype=np.int64)
