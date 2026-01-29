#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轻量化骨干网络模块
包含GhostNet、ShuffleNet等轻量化模块的实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def autopad(kernel_size: int, dilation: int = 1) -> int:
    """计算same填充大小"""
    return (kernel_size - 1) // 2 * dilation


class standard_conv(nn.Module):
    """
    标准卷积模块: Conv + BN + SiLU
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel: int = 1, 
        stride: int = 1, 
        padding: int = None, 
        group: int = 1,
        activation: bool = True
    ):
        """初始化标准卷积"""
        super().__init__()
        
        self.卷积 = nn.Conv2d(
            in_channels, out_channels, kernel, stride,
            autopad(kernel) if padding is None else padding,
            groups=group, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU(inplace=True) if activation else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.bn(self.卷积(x)))
    
    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """融合后的前向传播"""
        return self.activation(self.卷积(x))


class depthwise_separable_conv(nn.Module):
    """
    深度可分离卷积: Depthwise + Pointwise
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel: int = 3, stride: int = 1):
        """初始化深度可分离卷积"""
        super().__init__()
        
        # 深度卷积
        self.深度卷积 = nn.Conv2d(
            in_channels, in_channels, kernel, stride,
            autopad(kernel), groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        # 逐点卷积
        self.逐点卷积 = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.activation = nn.SiLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.bn1(self.深度卷积(x)))
        x = self.activation(self.bn2(self.逐点卷积(x)))
        return x


class GhostModule(nn.Module):
    """
    GhostNet的基础模块
    
    通过廉价操作生成更多特征图，减少计算量
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel: int = 1, 
        ratio: int = 2, 
        dw_kernel: int = 3,
        stride: int = 1, 
        activation: bool = True
    ):
        """
        初始化Ghost模块
        
        参数:
            input通道: input通道数
            output通道: output通道数
            卷积核: 主卷积核大小
            比例: Ghost比例
            dw卷积核: 深度卷积核大小
            步长: 步长
            激活: 是否使用激活函数
        """
        super().__init__()
        
        self.out_channels = out_channels
        初始通道 = math.ceil(out_channels / ratio)
        新通道 = 初始通道 * (ratio - 1)
        
        # 主卷积
        self.主卷积 = nn.Sequential(
            nn.Conv2d(in_channels, 初始通道, kernel, stride, autopad(kernel), bias=False),
            nn.BatchNorm2d(初始通道),
            nn.SiLU(inplace=True) if activation else nn.Identity(),
        )
        
        # 廉价操作 (深度卷积)
        self.廉价操作 = nn.Sequential(
            nn.Conv2d(初始通道, 新通道, dw_kernel, 1, autopad(dw_kernel), groups=初始通道, bias=False),
            nn.BatchNorm2d(新通道),
            nn.SiLU(inplace=True) if activation else nn.Identity(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        主特征 = self.主卷积(x)
        廉价特征 = self.廉价操作(主特征)
        output = torch.cat([主特征, 廉价特征], dim=1)
        return output[:, :self.out_channels, :, :]


class GhostBottleneck(nn.Module):
    """
    GhostNet瓶颈层
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        hidden_channels: int = None,
        kernel: int = 3, 
        stride: int = 1,
        use_se: bool = False
    ):
        """初始化Ghost瓶颈"""
        super().__init__()
        
        if hidden_channels is None:
            hidden_channels = out_channels
        
        self.stride = stride
        self.use_se = use_se
        
        # Ghost模块1
        self.ghost1 = GhostModule(in_channels, hidden_channels, kernel=1)
        
        # 深度卷积 (仅当步长>1时)
        if stride > 1:
            self.dw卷积 = nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, kernel, stride, autopad(kernel), groups=hidden_channels, bias=False),
                nn.BatchNorm2d(hidden_channels),
            )
        
        # SE模块
        if use_se:
            from .attention import SEAttention
            self.se = SEAttention(hidden_channels)
        
        # Ghost模块2
        self.ghost2 = GhostModule(hidden_channels, out_channels, kernel=1, activation=False)
        
        # 残差连接
        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel, stride, autopad(kernel), groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        残差 = self.shortcut(x)
        
        out = self.ghost1(x)
        if self.stride > 1:
            out = self.dw卷积(out)
        if self.use_se:
            out = self.se(out)
        out = self.ghost2(out)
        
        return out + 残差


class shuffle_channels(nn.Module):
    """
    通道重排操作，用于ShuffleNet
    """
    
    def __init__(self, groups: int):
        """初始化通道重排"""
        super().__init__()
        self.groups = groups
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        g = self.groups
        
        # 重排通道
        x = x.view(b, g, c // g, h, w)
        x = x.transpose(1, 2).contiguous()
        x = x.view(b, c, h, w)
        
        return x


class ShuffleNetUnit(nn.Module):
    """
    ShuffleNet V2 基本单元
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        """初始化ShuffleNet单元"""
        super().__init__()
        
        self.stride = stride
        
        if stride == 1:
            # 步长为1时，input通道分两半
            分支通道 = out_channels // 2
            
            self.分支1 = nn.Identity()
            self.分支2 = nn.Sequential(
                nn.Conv2d(分支通道, 分支通道, 1, 1, 0, bias=False),
                nn.BatchNorm2d(分支通道),
                nn.SiLU(inplace=True),
                nn.Conv2d(分支通道, 分支通道, 3, 1, 1, groups=分支通道, bias=False),
                nn.BatchNorm2d(分支通道),
                nn.Conv2d(分支通道, 分支通道, 1, 1, 0, bias=False),
                nn.BatchNorm2d(分支通道),
                nn.SiLU(inplace=True),
            )
        else:
            # 步长为2时，使用两个分支进行下采样
            self.分支1 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels // 2, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels // 2),
                nn.SiLU(inplace=True),
            )
            self.分支2 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels // 2, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels // 2),
                nn.SiLU(inplace=True),
                nn.Conv2d(out_channels // 2, out_channels // 2, 3, stride, 1, groups=out_channels // 2, bias=False),
                nn.BatchNorm2d(out_channels // 2),
                nn.Conv2d(out_channels // 2, out_channels // 2, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels // 2),
                nn.SiLU(inplace=True),
            )
        
        self.通道重排 = shuffle_channels(2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat([x1, self.分支2(x2)], dim=1)
        else:
            out = torch.cat([self.分支1(x), self.分支2(x)], dim=1)
        
        out = self.通道重排(out)
        return out


class GhostC3(nn.Module):
    """
    使用Ghost模块替换的C3模块
    """
    
    def __init__(self, in_channels: int, out_channels: int, count: int = 1, shortcut: bool = True):
        """初始化GhostC3"""
        super().__init__()
        
        hidden_channels = out_channels // 2
        
        self.cv1 = GhostModule(in_channels, hidden_channels)
        self.cv2 = GhostModule(in_channels, hidden_channels)
        self.cv3 = GhostModule(2 * hidden_channels, out_channels)
        
        self.m = nn.Sequential(
            *[GhostBottleneck(hidden_channels, hidden_channels) for _ in range(count)]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat([self.m(self.cv1(x)), self.cv2(x)], dim=1))


class ShuffleC3(nn.Module):
    """
    使用ShuffleNet单元替换的C3模块
    """
    
    def __init__(self, in_channels: int, out_channels: int, count: int = 1, shortcut: bool = True):
        """初始化ShuffleC3"""
        super().__init__()
        
        hidden_channels = out_channels // 2
        
        self.cv1 = standard_conv(in_channels, hidden_channels, 1)
        self.cv2 = standard_conv(in_channels, hidden_channels, 1)
        self.cv3 = standard_conv(2 * hidden_channels, out_channels, 1)
        
        self.m = nn.Sequential(
            *[ShuffleNetUnit(hidden_channels, hidden_channels, 1) for _ in range(count)]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat([self.m(self.cv1(x)), self.cv2(x)], dim=1))
