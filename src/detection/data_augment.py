#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
红外图像数据增强模块
针对红外图像特点设计的数据增强方法
"""

import numpy as np
import cv2
import random
from typing import Tuple, List, Optional, Dict


class 红外数据增强器:
    """
    红外图像数据增强器
    
    包含针对红外图像特点的增强方法
    """
    
    def __init__(
        self,
        亮度范围: Tuple[float, float] = (0.8, 1.2),
        对比度范围: Tuple[float, float] = (0.8, 1.2),
        噪声强度: float = 0.02,
        模糊概率: float = 0.3,
        翻转概率: float = 0.5,
        旋转角度: int = 10,
        缩放范围: Tuple[float, float] = (0.8, 1.2),
        裁剪范围: Tuple[float, float] = (0.8, 1.0),
        Mosaic概率: float = 0.5,
        MixUp概率: float = 0.3,
    ):
        """
        初始化数据增强器
        
        参数:
            亮度范围: 亮度调整范围
            对比度范围: 对比度调整范围
            噪声强度: 高斯噪声强度
            模糊概率: 应用模糊的概率
            翻转概率: 水平翻转概率
            旋转角度: 最大旋转角度
            缩放范围: 缩放范围
            裁剪范围: 随机裁剪范围
            Mosaic概率: Mosaic增强概率
            MixUp概率: MixUp增强概率
        """
        self.亮度范围 = 亮度范围
        self.对比度范围 = 对比度范围
        self.噪声强度 = 噪声强度
        self.模糊概率 = 模糊概率
        self.翻转概率 = 翻转概率
        self.旋转角度 = 旋转角度
        self.缩放范围 = 缩放范围
        self.裁剪范围 = 裁剪范围
        self.Mosaic概率 = Mosaic概率
        self.MixUp概率 = MixUp概率
    
    def __call__(
        self, 
        图像: np.ndarray, 
        标签: np.ndarray = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        应用数据增强
        
        参数:
            图像: 输入图像
            标签: 边界框标签 [N, 5] (class, x, y, w, h) 归一化坐标
        
        返回:
            增强后的图像和标签
        """
        # 基础增强
        图像, 标签 = self.随机亮度(图像, 标签)
        图像, 标签 = self.随机对比度(图像, 标签)
        
        # 红外特定增强
        图像, 标签 = self.添加热噪声(图像, 标签)
        图像, 标签 = self.模拟温度漂移(图像, 标签)
        
        # 几何变换
        if random.random() < self.翻转概率:
            图像, 标签 = self.水平翻转(图像, 标签)
        
        图像, 标签 = self.随机旋转(图像, 标签)
        
        # 模糊
        if random.random() < self.模糊概率:
            图像, 标签 = self.随机模糊(图像, 标签)
        
        return 图像, 标签
    
    def 随机亮度(
        self, 
        图像: np.ndarray, 
        标签: np.ndarray = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        随机亮度调整
        """
        系数 = random.uniform(self.亮度范围[0], self.亮度范围[1])
        图像 = np.clip(图像 * 系数, 0, 255).astype(np.uint8)
        return 图像, 标签
    
    def 随机对比度(
        self, 
        图像: np.ndarray, 
        标签: np.ndarray = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        随机对比度调整
        """
        系数 = random.uniform(self.对比度范围[0], self.对比度范围[1])
        均值 = np.mean(图像)
        图像 = np.clip((图像 - 均值) * 系数 + 均值, 0, 255).astype(np.uint8)
        return 图像, 标签
    
    def 添加热噪声(
        self, 
        图像: np.ndarray, 
        标签: np.ndarray = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        添加模拟红外热噪声
        
        红外传感器特有的热噪声，与温度相关
        """
        if self.噪声强度 <= 0:
            return 图像, 标签
        
        # 生成高斯噪声
        噪声 = np.random.randn(*图像.shape) * self.噪声强度 * 255
        图像 = np.clip(图像.astype(np.float32) + 噪声, 0, 255).astype(np.uint8)
        
        return 图像, 标签
    
    def 模拟温度漂移(
        self, 
        图像: np.ndarray, 
        标签: np.ndarray = None,
        强度: float = 0.1
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        模拟红外图像的温度漂移效应
        
        红外传感器会因环境温度变化产生整体亮度偏移
        """
        if random.random() > 0.3:
            return 图像, 标签
        
        偏移量 = random.uniform(-强度, 强度) * 255
        图像 = np.clip(图像.astype(np.float32) + 偏移量, 0, 255).astype(np.uint8)
        
        return 图像, 标签
    
    def 水平翻转(
        self, 
        图像: np.ndarray, 
        标签: np.ndarray = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        水平翻转
        """
        图像 = cv2.flip(图像, 1)
        
        if 标签 is not None and len(标签) > 0:
            标签 = 标签.copy()
            标签[:, 1] = 1 - 标签[:, 1]  # x中心坐标翻转
        
        return 图像, 标签
    
    def 随机旋转(
        self, 
        图像: np.ndarray, 
        标签: np.ndarray = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        随机旋转
        """
        if self.旋转角度 <= 0:
            return 图像, 标签
        
        角度 = random.uniform(-self.旋转角度, self.旋转角度)
        
        高, 宽 = 图像.shape[:2]
        中心 = (宽 / 2, 高 / 2)
        
        # 计算旋转矩阵
        旋转矩阵 = cv2.getRotationMatrix2D(中心, 角度, 1.0)
        
        # 应用旋转
        图像 = cv2.warpAffine(图像, 旋转矩阵, (宽, 高), borderValue=(114, 114, 114))
        
        # 更新标签坐标
        if 标签 is not None and len(标签) > 0:
            标签 = self._旋转标签(标签, 旋转矩阵, 宽, 高)
        
        return 图像, 标签
    
    def _旋转标签(
        self, 
        标签: np.ndarray, 
        旋转矩阵: np.ndarray,
        图像宽: int,
        图像高: int
    ) -> np.ndarray:
        """
        旋转标签坐标
        """
        新标签 = []
        
        for 标注 in 标签:
            类别, x_c, y_c, w, h = 标注
            
            # 转换为像素坐标
            x_c_px = x_c * 图像宽
            y_c_px = y_c * 图像高
            
            # 应用旋转
            点 = np.array([[x_c_px, y_c_px, 1]])
            新点 = 旋转矩阵.dot(点.T).T[0]
            
            # 转回归一化坐标
            新_x_c = 新点[0] / 图像宽
            新_y_c = 新点[1] / 图像高
            
            # 检查有效性
            if 0 <= 新_x_c <= 1 and 0 <= 新_y_c <= 1:
                新标签.append([类别, 新_x_c, 新_y_c, w, h])
        
        return np.array(新标签) if 新标签 else np.array([]).reshape(0, 5)
    
    def 随机模糊(
        self, 
        图像: np.ndarray, 
        标签: np.ndarray = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        随机模糊
        """
        模糊类型 = random.choice(['gaussian', 'median', 'motion'])
        
        if 模糊类型 == 'gaussian':
            核大小 = random.choice([3, 5])
            图像 = cv2.GaussianBlur(图像, (核大小, 核大小), 0)
        elif 模糊类型 == 'median':
            核大小 = random.choice([3, 5])
            图像 = cv2.medianBlur(图像, 核大小)
        elif 模糊类型 == 'motion':
            图像 = self._运动模糊(图像)
        
        return 图像, 标签
    
    def _运动模糊(self, 图像: np.ndarray, 强度: int = 15) -> np.ndarray:
        """
        运动模糊效果
        """
        角度 = random.randint(0, 360)
        核大小 = random.randint(5, 强度)
        
        # 创建运动模糊核
        核 = np.zeros((核大小, 核大小))
        核[核大小 // 2, :] = 1
        
        # 旋转核
        中心 = (核大小 / 2, 核大小 / 2)
        旋转矩阵 = cv2.getRotationMatrix2D(中心, 角度, 1.0)
        核 = cv2.warpAffine(核, 旋转矩阵, (核大小, 核大小))
        核 /= 核.sum()
        
        # 应用模糊
        图像 = cv2.filter2D(图像, -1, 核)
        
        return 图像
    
    def 随机裁剪(
        self, 
        图像: np.ndarray, 
        标签: np.ndarray = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        随机裁剪
        """
        高, 宽 = 图像.shape[:2]
        
        裁剪比例 = random.uniform(self.裁剪范围[0], self.裁剪范围[1])
        新宽 = int(宽 * 裁剪比例)
        新高 = int(高 * 裁剪比例)
        
        # 随机裁剪位置
        x1 = random.randint(0, 宽 - 新宽)
        y1 = random.randint(0, 高 - 新高)
        
        图像 = 图像[y1:y1+新高, x1:x1+新宽]
        
        # 调整标签
        if 标签 is not None and len(标签) > 0:
            标签 = 标签.copy()
            
            # 转换坐标
            标签[:, 1] = (标签[:, 1] * 宽 - x1) / 新宽
            标签[:, 2] = (标签[:, 2] * 高 - y1) / 新高
            标签[:, 3] = 标签[:, 3] * 宽 / 新宽
            标签[:, 4] = 标签[:, 4] * 高 / 新高
            
            # 过滤出界的框
            有效掩码 = (
                (标签[:, 1] > 0) & (标签[:, 1] < 1) &
                (标签[:, 2] > 0) & (标签[:, 2] < 1)
            )
            标签 = 标签[有效掩码]
        
        return 图像, 标签
    
    def Mosaic增强(
        self, 
        图像列表: List[np.ndarray], 
        标签列表: List[np.ndarray],
        输出尺寸: int = 640
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mosaic数据增强
        
        将4张图像拼接成一张
        """
        if len(图像列表) < 4:
            raise ValueError("Mosaic需要至少4张图像")
        
        # 随机选择4张图像
        选中索引 = random.sample(range(len(图像列表)), 4)
        选中图像 = [图像列表[i] for i in 选中索引]
        选中标签 = [标签列表[i] for i in 选中索引]
        
        # 创建输出图像
        输出图像 = np.full((输出尺寸, 输出尺寸, 3), 114, dtype=np.uint8)
        
        # 随机中心点
        中心x = int(random.uniform(输出尺寸 * 0.25, 输出尺寸 * 0.75))
        中心y = int(random.uniform(输出尺寸 * 0.25, 输出尺寸 * 0.75))
        
        输出标签列表 = []
        
        # 放置4张图像
        for i, (图像, 标签) in enumerate(zip(选中图像, 选中标签)):
            高, 宽 = 图像.shape[:2]
            
            if i == 0:  # 左上
                x1a, y1a = max(中心x - 宽, 0), max(中心y - 高, 0)
                x2a, y2a = 中心x, 中心y
            elif i == 1:  # 右上
                x1a, y1a = 中心x, max(中心y - 高, 0)
                x2a, y2a = min(中心x + 宽, 输出尺寸), 中心y
            elif i == 2:  # 左下
                x1a, y1a = max(中心x - 宽, 0), 中心y
                x2a, y2a = 中心x, min(中心y + 高, 输出尺寸)
            else:  # 右下
                x1a, y1a = 中心x, 中心y
                x2a, y2a = min(中心x + 宽, 输出尺寸), min(中心y + 高, 输出尺寸)
            
            # 调整图像大小并放置
            区域宽 = x2a - x1a
            区域高 = y2a - y1a
            
            if 区域宽 > 0 and 区域高 > 0:
                调整图像 = cv2.resize(图像, (区域宽, 区域高))
                输出图像[y1a:y2a, x1a:x2a] = 调整图像
                
                # 调整标签
                if 标签 is not None and len(标签) > 0:
                    调整标签 = 标签.copy()
                    调整标签[:, 1] = (调整标签[:, 1] * 区域宽 + x1a) / 输出尺寸
                    调整标签[:, 2] = (调整标签[:, 2] * 区域高 + y1a) / 输出尺寸
                    调整标签[:, 3] = 调整标签[:, 3] * 区域宽 / 输出尺寸
                    调整标签[:, 4] = 调整标签[:, 4] * 区域高 / 输出尺寸
                    输出标签列表.append(调整标签)
        
        # 合并标签
        if 输出标签列表:
            输出标签 = np.concatenate(输出标签列表, axis=0)
        else:
            输出标签 = np.array([]).reshape(0, 5)
        
        return 输出图像, 输出标签
    
    def MixUp增强(
        self, 
        图像1: np.ndarray, 
        标签1: np.ndarray,
        图像2: np.ndarray, 
        标签2: np.ndarray,
        alpha: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        MixUp数据增强
        
        将两张图像按比例混合
        """
        # 调整到相同大小
        高1, 宽1 = 图像1.shape[:2]
        图像2 = cv2.resize(图像2, (宽1, 高1))
        
        # 随机混合系数
        混合系数 = random.beta(alpha, alpha)
        
        # 混合图像
        混合图像 = (图像1 * 混合系数 + 图像2 * (1 - 混合系数)).astype(np.uint8)
        
        # 合并标签
        if 标签1 is not None and 标签2 is not None:
            混合标签 = np.concatenate([标签1, 标签2], axis=0) if len(标签1) > 0 or len(标签2) > 0 else np.array([]).reshape(0, 5)
        else:
            混合标签 = 标签1 if 标签1 is not None else 标签2
        
        return 混合图像, 混合标签


def 创建训练增强器() -> 红外数据增强器:
    """创建训练用数据增强器"""
    return 红外数据增强器(
        亮度范围=(0.7, 1.3),
        对比度范围=(0.7, 1.3),
        噪声强度=0.03,
        模糊概率=0.3,
        翻转概率=0.5,
        旋转角度=15,
        Mosaic概率=0.5,
        MixUp概率=0.2,
    )


def 创建验证增强器() -> 红外数据增强器:
    """创建验证用数据增强器（不进行增强）"""
    return 红外数据增强器(
        亮度范围=(1.0, 1.0),
        对比度范围=(1.0, 1.0),
        噪声强度=0,
        模糊概率=0,
        翻转概率=0,
        旋转角度=0,
        Mosaic概率=0,
        MixUp概率=0,
    )
