#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
注意力机制模块
包含CBAM、SE、CoordAttention等注意力机制的实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEAttention(nn.Module):
    """
    Squeeze-and-Excitation注意力模块
    
    通过全局平均池化和全连接层学习通道间的依赖关系
    """
    
    def __init__(self, channels: int, reduction_ratio: int = 16):
        """
        初始化SE模块
        
        参数:
            通道数: input通道数
            缩减比例: 中间层通道缩减比例
        """
        super().__init__()
        
        mid_channels = max(channels // reduction_ratio, 8)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        b, c, _, _ = x.size()
        
        # 全局平均池化
        y = self.global_pool(x).view(b, c)
        
        # 全连接层
        y = self.fc(y).view(b, c, 1, 1)
        
        # 通道加权
        return x * y.expand_as(x)


class ChannelAttention(nn.Module):
    """
    CBAM中的通道注意力模块
    """
    
    def __init__(self, channels: int, reduction_ratio: int = 16):
        """
        初始化通道注意力模块
        
        参数:
            通道数: input通道数
            缩减比例: MLP缩减比例
        """
        super().__init__()
        
        mid_channels = max(channels // reduction_ratio, 8)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        avg_feat = self.shared_mlp(self.avg_pool(x))
        max_feat = self.shared_mlp(self.max_pool(x))
        
        attention = self.sigmoid(avg_feat + max_feat)
        
        return x * attention


class SpatialAttention(nn.Module):
    """
    CBAM中的空间注意力模块
    """
    
    def __init__(self, kernel_size: int = 7):
        """
        初始化空间注意力模块
        
        参数:
            卷积核大小: 卷积核大小
        """
        super().__init__()
        
        padding = (kernel_size - 1) // 2
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 通道维度池化
        avg_feat = torch.mean(x, dim=1, keepdim=True)
        max_feat, _ = torch.max(x, dim=1, keepdim=True)
        
        # 拼接
        feat = torch.cat([avg_feat, max_feat], dim=1)
        
        # 卷积生成注意力图
        attention = self.sigmoid(self.conv(feat))
        
        return x * attention


class CBAMAttention(nn.Module):
    """
    CBAM (Convolutional Block Attention Module) 注意力模块
    
    结合通道注意力和空间注意力
    """
    
    def __init__(self, channels: int, reduction_ratio: int = 16, kernel_size: int = 7):
        """
        初始化CBAM模块
        
        参数:
            通道数: input通道数
            缩减比例: 通道注意力缩减比例
            卷积核大小: 空间注意力卷积核大小
        """
        super().__init__()
        
        self.ChannelAttention = ChannelAttention(channels, reduction_ratio)
        self.SpatialAttention = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.ChannelAttention(x)
        x = self.SpatialAttention(x)
        return x


class CoordAttention(nn.Module):
    """
    Coordinate Attention (CoordAttention) 模块
    
    将位置信息编码到通道注意力中
    """
    
    def __init__(self, in_channels: int, out_channels: int, reduction_ratio: int = 32):
        """
        初始化坐标注意力模块
        
        参数:
            input通道: input通道数
            output通道: output通道数
            缩减比例: 通道缩减比例
        """
        super().__init__()
        
        mid_channels = max(8, in_channels // reduction_ratio)
        
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.activation = nn.ReLU(inplace=True)
        
        self.conv_h = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.conv_w = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        b, c, h, w = x.size()
        
        # 水平和垂直方向池化
        x_h = self.pool_h(x)  # [b, c, h, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [b, c, w, 1] -> [b, c, 1, w] -> permute -> [b, c, w, 1]
        
        # 拼接并编码
        y = torch.cat([x_h, x_w], dim=2)  # [b, c, h+w, 1]
        y = self.activation(self.bn1(self.conv1(y)))
        
        # 分割
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        # 生成注意力
        a_h = self.sigmoid(self.conv_h(x_h))
        a_w = self.sigmoid(self.conv_w(x_w))
        
        # 应用注意力
        return x * a_h * a_w


class ECAAttention(nn.Module):
    """
    ECA (Efficient Channel Attention) 模块
    
    使用一维卷积避免降维，更高效
    """
    
    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        """
        初始化ECA模块
        
        参数:
            通道数: input通道数
            gamma: 卷积核大小计算参数
            b: 卷积核大小计算参数
        """
        super().__init__()
        
        import math
        # 自适应确定卷积核大小
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 全局平均池化
        y = self.global_pool(x)  # [b, c, 1, 1]
        
        # 一维卷积
        y = y.squeeze(-1).transpose(-1, -2)  # [b, 1, c]
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)  # [b, c, 1, 1]
        
        # 注意力加权
        y = self.sigmoid(y)
        return x * y.expand_as(x)


def get_attention_module(name: str, channels: int, **kwargs) -> nn.Module:
    """
    根据name获取注意力模块
    
    参数:
        name: 注意力模块name (se/cbam/coordatt/eca)
        通道数: input通道数
        **kwargs: 额外参数
    
    返回:
        注意力模块实例
    """
    name = name.lower()
    
    if name == 'se':
        return SEAttention(channels, **kwargs)
    elif name == 'cbam':
        return CBAMAttention(channels, **kwargs)
    elif name in ['coordatt', 'coord', 'ca']:
        return CoordAttention(channels, channels, **kwargs)
    elif name == 'eca':
        return ECAAttention(channels, **kwargs)
    else:
        raise ValueError(f"不支持的注意力模块: {name}")
