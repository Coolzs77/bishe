#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
注意力机制模块
包含CBAM、SE、CoordAttention等注意力机制的实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SE注意力(nn.Module):
    """
    Squeeze-and-Excitation注意力模块
    
    通过全局平均池化和全连接层学习通道间的依赖关系
    """
    
    def __init__(self, 通道数: int, 缩减比例: int = 16):
        """
        初始化SE模块
        
        参数:
            通道数: 输入通道数
            缩减比例: 中间层通道缩减比例
        """
        super().__init__()
        
        中间通道 = max(通道数 // 缩减比例, 8)
        
        self.全局池化 = nn.AdaptiveAvgPool2d(1)
        self.全连接 = nn.Sequential(
            nn.Linear(通道数, 中间通道, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(中间通道, 通道数, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        b, c, _, _ = x.size()
        
        # 全局平均池化
        y = self.全局池化(x).view(b, c)
        
        # 全连接层
        y = self.全连接(y).view(b, c, 1, 1)
        
        # 通道加权
        return x * y.expand_as(x)


class 通道注意力(nn.Module):
    """
    CBAM中的通道注意力模块
    """
    
    def __init__(self, 通道数: int, 缩减比例: int = 16):
        """
        初始化通道注意力模块
        
        参数:
            通道数: 输入通道数
            缩减比例: MLP缩减比例
        """
        super().__init__()
        
        中间通道 = max(通道数 // 缩减比例, 8)
        
        self.平均池化 = nn.AdaptiveAvgPool2d(1)
        self.最大池化 = nn.AdaptiveMaxPool2d(1)
        
        self.共享MLP = nn.Sequential(
            nn.Conv2d(通道数, 中间通道, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(中间通道, 通道数, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        平均特征 = self.共享MLP(self.平均池化(x))
        最大特征 = self.共享MLP(self.最大池化(x))
        
        注意力 = self.sigmoid(平均特征 + 最大特征)
        
        return x * 注意力


class 空间注意力(nn.Module):
    """
    CBAM中的空间注意力模块
    """
    
    def __init__(self, 卷积核大小: int = 7):
        """
        初始化空间注意力模块
        
        参数:
            卷积核大小: 卷积核大小
        """
        super().__init__()
        
        填充 = (卷积核大小 - 1) // 2
        
        self.卷积 = nn.Conv2d(2, 1, 卷积核大小, padding=填充, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 通道维度池化
        平均特征 = torch.mean(x, dim=1, keepdim=True)
        最大特征, _ = torch.max(x, dim=1, keepdim=True)
        
        # 拼接
        特征 = torch.cat([平均特征, 最大特征], dim=1)
        
        # 卷积生成注意力图
        注意力 = self.sigmoid(self.卷积(特征))
        
        return x * 注意力


class CBAM注意力(nn.Module):
    """
    CBAM (Convolutional Block Attention Module) 注意力模块
    
    结合通道注意力和空间注意力
    """
    
    def __init__(self, 通道数: int, 缩减比例: int = 16, 卷积核大小: int = 7):
        """
        初始化CBAM模块
        
        参数:
            通道数: 输入通道数
            缩减比例: 通道注意力缩减比例
            卷积核大小: 空间注意力卷积核大小
        """
        super().__init__()
        
        self.通道注意力 = 通道注意力(通道数, 缩减比例)
        self.空间注意力 = 空间注意力(卷积核大小)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.通道注意力(x)
        x = self.空间注意力(x)
        return x


class 坐标注意力(nn.Module):
    """
    Coordinate Attention (CoordAttention) 模块
    
    将位置信息编码到通道注意力中
    """
    
    def __init__(self, 输入通道: int, 输出通道: int, 缩减比例: int = 32):
        """
        初始化坐标注意力模块
        
        参数:
            输入通道: 输入通道数
            输出通道: 输出通道数
            缩减比例: 通道缩减比例
        """
        super().__init__()
        
        中间通道 = max(8, 输入通道 // 缩减比例)
        
        self.水平池化 = nn.AdaptiveAvgPool2d((None, 1))
        self.垂直池化 = nn.AdaptiveAvgPool2d((1, None))
        
        self.卷积1 = nn.Conv2d(输入通道, 中间通道, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(中间通道)
        self.激活 = nn.ReLU(inplace=True)
        
        self.卷积_h = nn.Conv2d(中间通道, 输出通道, 1, bias=False)
        self.卷积_w = nn.Conv2d(中间通道, 输出通道, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        b, c, h, w = x.size()
        
        # 水平和垂直方向池化
        x_h = self.水平池化(x)  # [b, c, h, 1]
        x_w = self.垂直池化(x).permute(0, 1, 3, 2)  # [b, c, w, 1] -> [b, c, 1, w] -> permute -> [b, c, w, 1]
        
        # 拼接并编码
        y = torch.cat([x_h, x_w], dim=2)  # [b, c, h+w, 1]
        y = self.激活(self.bn1(self.卷积1(y)))
        
        # 分割
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        # 生成注意力
        a_h = self.sigmoid(self.卷积_h(x_h))
        a_w = self.sigmoid(self.卷积_w(x_w))
        
        # 应用注意力
        return x * a_h * a_w


class ECA注意力(nn.Module):
    """
    ECA (Efficient Channel Attention) 模块
    
    使用一维卷积避免降维，更高效
    """
    
    def __init__(self, 通道数: int, gamma: int = 2, b: int = 1):
        """
        初始化ECA模块
        
        参数:
            通道数: 输入通道数
            gamma: 卷积核大小计算参数
            b: 卷积核大小计算参数
        """
        super().__init__()
        
        import math
        # 自适应确定卷积核大小
        t = int(abs((math.log2(通道数) + b) / gamma))
        k = t if t % 2 else t + 1
        
        self.全局池化 = nn.AdaptiveAvgPool2d(1)
        self.卷积 = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 全局平均池化
        y = self.全局池化(x)  # [b, c, 1, 1]
        
        # 一维卷积
        y = y.squeeze(-1).transpose(-1, -2)  # [b, 1, c]
        y = self.卷积(y)
        y = y.transpose(-1, -2).unsqueeze(-1)  # [b, c, 1, 1]
        
        # 注意力加权
        y = self.sigmoid(y)
        return x * y.expand_as(x)


def 获取注意力模块(名称: str, 通道数: int, **kwargs) -> nn.Module:
    """
    根据名称获取注意力模块
    
    参数:
        名称: 注意力模块名称 (se/cbam/coordatt/eca)
        通道数: 输入通道数
        **kwargs: 额外参数
    
    返回:
        注意力模块实例
    """
    名称 = 名称.lower()
    
    if 名称 == 'se':
        return SE注意力(通道数, **kwargs)
    elif 名称 == 'cbam':
        return CBAM注意力(通道数, **kwargs)
    elif 名称 in ['coordatt', 'coord', 'ca']:
        return 坐标注意力(通道数, 通道数, **kwargs)
    elif 名称 == 'eca':
        return ECA注意力(通道数, **kwargs)
    else:
        raise ValueError(f"不支持的注意力模块: {名称}")
