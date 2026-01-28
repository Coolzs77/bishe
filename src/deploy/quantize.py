#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型量化模块
提供INT8量化相关功能
"""

import numpy as np
from pathlib import Path
from typing import List, Callable, Optional
import json


class 量化校准器:
    """
    模型量化校准器
    
    用于准备INT8量化所需的校准数据
    """
    
    def __init__(
        self,
        校准数据目录: str,
        输入尺寸: int = 640,
        样本数: int = 100,
        预处理函数: Callable = None
    ):
        """
        初始化校准器
        
        参数:
            校准数据目录: 校准图像目录
            输入尺寸: 输入图像尺寸
            样本数: 校准样本数量
            预处理函数: 自定义预处理函数
        """
        self.校准数据目录 = Path(校准数据目录)
        self.输入尺寸 = 输入尺寸
        self.样本数 = 样本数
        self.预处理函数 = 预处理函数 or self.默认预处理
        
        self.图像列表 = []
        self._收集图像()
    
    def _收集图像(self):
        """收集校准图像"""
        if not self.校准数据目录.exists():
            print(f"警告: 校准数据目录不存在 - {self.校准数据目录}")
            return
        
        for 后缀 in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.图像列表.extend(self.校准数据目录.glob(后缀))
        
        if len(self.图像列表) > self.样本数:
            import random
            self.图像列表 = random.sample(self.图像列表, self.样本数)
        
        print(f"收集到 {len(self.图像列表)} 张校准图像")
    
    def 默认预处理(self, 图像路径: str) -> np.ndarray:
        """
        默认预处理函数
        
        参数:
            图像路径: 图像文件路径
        
        返回:
            预处理后的图像数组
        """
        import cv2
        
        图像 = cv2.imread(str(图像路径))
        if 图像 is None:
            return None
        
        # 调整大小
        图像 = cv2.resize(图像, (self.输入尺寸, self.输入尺寸))
        
        # BGR转RGB
        图像 = cv2.cvtColor(图像, cv2.COLOR_BGR2RGB)
        
        # 归一化
        图像 = 图像.astype(np.float32) / 255.0
        
        # HWC转CHW
        图像 = np.transpose(图像, (2, 0, 1))
        
        return 图像
    
    def 生成校准数据(self) -> List[np.ndarray]:
        """
        生成校准数据
        
        返回:
            校准数据数组列表
        """
        校准数据 = []
        
        for 图像路径 in self.图像列表:
            数据 = self.预处理函数(图像路径)
            if 数据 is not None:
                校准数据.append(数据)
        
        print(f"生成 {len(校准数据)} 个校准样本")
        return 校准数据
    
    def 保存校准列表(self, 输出路径: str):
        """
        保存校准图像列表
        
        参数:
            输出路径: 输出文件路径
        """
        with open(输出路径, 'w') as f:
            for 图像路径 in self.图像列表:
                f.write(str(图像路径.absolute()) + '\n')
        
        print(f"校准列表已保存: {输出路径}")


def 计算量化参数(
    数据: np.ndarray,
    量化位数: int = 8,
    对称量化: bool = True
) -> dict:
    """
    计算量化参数
    
    参数:
        数据: 输入数据数组
        量化位数: 量化位数
        对称量化: 是否使用对称量化
    
    返回:
        量化参数字典
    """
    最大值 = np.max(np.abs(数据))
    最小值 = np.min(数据)
    最大数据 = np.max(数据)
    
    if 对称量化:
        # 对称量化
        范围 = 2 ** (量化位数 - 1) - 1
        缩放因子 = 最大值 / 范围
        零点 = 0
    else:
        # 非对称量化
        范围 = 2 ** 量化位数 - 1
        缩放因子 = (最大数据 - 最小值) / 范围
        零点 = round(-最小值 / 缩放因子)
    
    return {
        '缩放因子': float(缩放因子),
        '零点': int(零点),
        '最小值': float(最小值),
        '最大值': float(最大数据),
        '量化位数': 量化位数,
        '对称量化': 对称量化,
    }


def 量化张量(
    数据: np.ndarray,
    缩放因子: float,
    零点: int,
    量化位数: int = 8,
    有符号: bool = True
) -> np.ndarray:
    """
    量化张量
    
    参数:
        数据: 输入浮点数据
        缩放因子: 量化缩放因子
        零点: 量化零点
        量化位数: 量化位数
        有符号: 是否为有符号整数
    
    返回:
        量化后的整数数组
    """
    # 量化
    量化数据 = np.round(数据 / 缩放因子) + 零点
    
    # 裁剪到有效范围
    if 有符号:
        最小 = -(2 ** (量化位数 - 1))
        最大 = 2 ** (量化位数 - 1) - 1
        dtype = np.int8
    else:
        最小 = 0
        最大 = 2 ** 量化位数 - 1
        dtype = np.uint8
    
    量化数据 = np.clip(量化数据, 最小, 最大).astype(dtype)
    
    return 量化数据


def 反量化张量(
    量化数据: np.ndarray,
    缩放因子: float,
    零点: int
) -> np.ndarray:
    """
    反量化张量
    
    参数:
        量化数据: 量化后的整数数据
        缩放因子: 量化缩放因子
        零点: 量化零点
    
    返回:
        反量化后的浮点数组
    """
    return (量化数据.astype(np.float32) - 零点) * 缩放因子


def 评估量化误差(
    原始数据: np.ndarray,
    量化数据: np.ndarray,
    缩放因子: float,
    零点: int
) -> dict:
    """
    评估量化误差
    
    参数:
        原始数据: 原始浮点数据
        量化数据: 量化后的整数数据
        缩放因子: 量化缩放因子
        零点: 量化零点
    
    返回:
        误差统计字典
    """
    # 反量化
    重建数据 = 反量化张量(量化数据, 缩放因子, 零点)
    
    # 计算误差
    绝对误差 = np.abs(原始数据 - 重建数据)
    相对误差 = 绝对误差 / (np.abs(原始数据) + 1e-10)
    
    return {
        '平均绝对误差': float(np.mean(绝对误差)),
        '最大绝对误差': float(np.max(绝对误差)),
        '平均相对误差': float(np.mean(相对误差)),
        '最大相对误差': float(np.max(相对误差)),
        'RMSE': float(np.sqrt(np.mean((原始数据 - 重建数据) ** 2))),
    }


class 模型量化器:
    """
    模型量化器类
    """
    
    def __init__(
        self,
        量化位数: int = 8,
        对称量化: bool = True
    ):
        """
        初始化量化器
        
        参数:
            量化位数: 量化位数
            对称量化: 是否使用对称量化
        """
        self.量化位数 = 量化位数
        self.对称量化 = 对称量化
        self.量化参数 = {}
    
    def 校准(self, 层名称: str, 数据: np.ndarray):
        """
        校准某一层的量化参数
        
        参数:
            层名称: 层名称
            数据: 该层的激活数据
        """
        参数 = 计算量化参数(数据, self.量化位数, self.对称量化)
        self.量化参数[层名称] = 参数
        
        print(f"  {层名称}: scale={参数['缩放因子']:.6f}, zp={参数['零点']}")
    
    def 量化层(self, 层名称: str, 数据: np.ndarray) -> np.ndarray:
        """
        量化某一层的数据
        
        参数:
            层名称: 层名称
            数据: 浮点数据
        
        返回:
            量化后的数据
        """
        if 层名称 not in self.量化参数:
            raise ValueError(f"层 {层名称} 未校准")
        
        参数 = self.量化参数[层名称]
        return 量化张量(
            数据, 
            参数['缩放因子'], 
            参数['零点'],
            self.量化位数
        )
    
    def 保存量化参数(self, 输出路径: str):
        """
        保存量化参数
        
        参数:
            输出路径: 输出文件路径
        """
        with open(输出路径, 'w', encoding='utf-8') as f:
            json.dump(self.量化参数, f, indent=2, ensure_ascii=False)
        
        print(f"量化参数已保存: {输出路径}")
    
    def 加载量化参数(self, 输入路径: str):
        """
        加载量化参数
        
        参数:
            输入路径: 输入文件路径
        """
        with open(输入路径, 'r', encoding='utf-8') as f:
            self.量化参数 = json.load(f)
        
        print(f"量化参数已加载: {输入路径}")
