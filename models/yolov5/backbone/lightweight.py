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


def 自动填充(卷积核大小: int, 膨胀率: int = 1) -> int:
    """计算same填充大小"""
    return (卷积核大小 - 1) // 2 * 膨胀率


class 标准卷积(nn.Module):
    """
    标准卷积模块: Conv + BN + SiLU
    """
    
    def __init__(
        self, 
        input通道: int, 
        output通道: int, 
        卷积核: int = 1, 
        步长: int = 1, 
        填充: int = None, 
        分组: int = 1,
        激活: bool = True
    ):
        """初始化标准卷积"""
        super().__init__()
        
        self.卷积 = nn.Conv2d(
            input通道, output通道, 卷积核, 步长,
            自动填充(卷积核) if 填充 is None else 填充,
            groups=分组, bias=False
        )
        self.bn = nn.BatchNorm2d(output通道)
        self.激活 = nn.SiLU(inplace=True) if 激活 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.激活(self.bn(self.卷积(x)))
    
    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """融合后的前向传播"""
        return self.激活(self.卷积(x))


class 深度可分离卷积(nn.Module):
    """
    深度可分离卷积: Depthwise + Pointwise
    """
    
    def __init__(self, input通道: int, output通道: int, 卷积核: int = 3, 步长: int = 1):
        """初始化深度可分离卷积"""
        super().__init__()
        
        # 深度卷积
        self.深度卷积 = nn.Conv2d(
            input通道, input通道, 卷积核, 步长,
            自动填充(卷积核), groups=input通道, bias=False
        )
        self.bn1 = nn.BatchNorm2d(input通道)
        
        # 逐点卷积
        self.逐点卷积 = nn.Conv2d(input通道, output通道, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(output通道)
        
        self.激活 = nn.SiLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.激活(self.bn1(self.深度卷积(x)))
        x = self.激活(self.bn2(self.逐点卷积(x)))
        return x


class Ghost模块(nn.Module):
    """
    GhostNet的基础模块
    
    通过廉价操作生成更多特征图，减少计算量
    """
    
    def __init__(
        self, 
        input通道: int, 
        output通道: int, 
        卷积核: int = 1, 
        比例: int = 2, 
        dw卷积核: int = 3,
        步长: int = 1, 
        激活: bool = True
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
        
        self.output通道 = output通道
        初始通道 = math.ceil(output通道 / 比例)
        新通道 = 初始通道 * (比例 - 1)
        
        # 主卷积
        self.主卷积 = nn.Sequential(
            nn.Conv2d(input通道, 初始通道, 卷积核, 步长, 自动填充(卷积核), bias=False),
            nn.BatchNorm2d(初始通道),
            nn.SiLU(inplace=True) if 激活 else nn.Identity(),
        )
        
        # 廉价操作 (深度卷积)
        self.廉价操作 = nn.Sequential(
            nn.Conv2d(初始通道, 新通道, dw卷积核, 1, 自动填充(dw卷积核), groups=初始通道, bias=False),
            nn.BatchNorm2d(新通道),
            nn.SiLU(inplace=True) if 激活 else nn.Identity(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        主特征 = self.主卷积(x)
        廉价特征 = self.廉价操作(主特征)
        output = torch.cat([主特征, 廉价特征], dim=1)
        return output[:, :self.output通道, :, :]


class Ghost瓶颈(nn.Module):
    """
    GhostNet瓶颈层
    """
    
    def __init__(
        self, 
        input通道: int, 
        output通道: int, 
        隐藏通道: int = None,
        卷积核: int = 3, 
        步长: int = 1,
        使用SE: bool = False
    ):
        """初始化Ghost瓶颈"""
        super().__init__()
        
        if 隐藏通道 is None:
            隐藏通道 = output通道
        
        self.步长 = 步长
        self.使用SE = 使用SE
        
        # Ghost模块1
        self.ghost1 = Ghost模块(input通道, 隐藏通道, 卷积核=1)
        
        # 深度卷积 (仅当步长>1时)
        if 步长 > 1:
            self.dw卷积 = nn.Sequential(
                nn.Conv2d(隐藏通道, 隐藏通道, 卷积核, 步长, 自动填充(卷积核), groups=隐藏通道, bias=False),
                nn.BatchNorm2d(隐藏通道),
            )
        
        # SE模块
        if 使用SE:
            from .attention import SE注意力
            self.se = SE注意力(隐藏通道)
        
        # Ghost模块2
        self.ghost2 = Ghost模块(隐藏通道, output通道, 卷积核=1, 激活=False)
        
        # 残差连接
        if 步长 == 1 and input通道 == output通道:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input通道, input通道, 卷积核, 步长, 自动填充(卷积核), groups=input通道, bias=False),
                nn.BatchNorm2d(input通道),
                nn.Conv2d(input通道, output通道, 1, 1, 0, bias=False),
                nn.BatchNorm2d(output通道),
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        残差 = self.shortcut(x)
        
        out = self.ghost1(x)
        if self.步长 > 1:
            out = self.dw卷积(out)
        if self.使用SE:
            out = self.se(out)
        out = self.ghost2(out)
        
        return out + 残差


class Shuffle通道(nn.Module):
    """
    通道重排操作，用于ShuffleNet
    """
    
    def __init__(self, 分组数: int):
        """初始化通道重排"""
        super().__init__()
        self.分组数 = 分组数
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        g = self.分组数
        
        # 重排通道
        x = x.view(b, g, c // g, h, w)
        x = x.transpose(1, 2).contiguous()
        x = x.view(b, c, h, w)
        
        return x


class ShuffleNet单元(nn.Module):
    """
    ShuffleNet V2 基本单元
    """
    
    def __init__(self, input通道: int, output通道: int, 步长: int = 1):
        """初始化ShuffleNet单元"""
        super().__init__()
        
        self.步长 = 步长
        
        if 步长 == 1:
            # 步长为1时，input通道分两半
            分支通道 = output通道 // 2
            
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
                nn.Conv2d(input通道, input通道, 3, 步长, 1, groups=input通道, bias=False),
                nn.BatchNorm2d(input通道),
                nn.Conv2d(input通道, output通道 // 2, 1, 1, 0, bias=False),
                nn.BatchNorm2d(output通道 // 2),
                nn.SiLU(inplace=True),
            )
            self.分支2 = nn.Sequential(
                nn.Conv2d(input通道, output通道 // 2, 1, 1, 0, bias=False),
                nn.BatchNorm2d(output通道 // 2),
                nn.SiLU(inplace=True),
                nn.Conv2d(output通道 // 2, output通道 // 2, 3, 步长, 1, groups=output通道 // 2, bias=False),
                nn.BatchNorm2d(output通道 // 2),
                nn.Conv2d(output通道 // 2, output通道 // 2, 1, 1, 0, bias=False),
                nn.BatchNorm2d(output通道 // 2),
                nn.SiLU(inplace=True),
            )
        
        self.通道重排 = Shuffle通道(2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.步长 == 1:
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
    
    def __init__(self, input通道: int, output通道: int, count: int = 1, shortcut: bool = True):
        """初始化GhostC3"""
        super().__init__()
        
        隐藏通道 = output通道 // 2
        
        self.cv1 = Ghost模块(input通道, 隐藏通道)
        self.cv2 = Ghost模块(input通道, 隐藏通道)
        self.cv3 = Ghost模块(2 * 隐藏通道, output通道)
        
        self.m = nn.Sequential(
            *[Ghost瓶颈(隐藏通道, 隐藏通道) for _ in range(count)]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat([self.m(self.cv1(x)), self.cv2(x)], dim=1))


class ShuffleC3(nn.Module):
    """
    使用ShuffleNet单元替换的C3模块
    """
    
    def __init__(self, input通道: int, output通道: int, count: int = 1, shortcut: bool = True):
        """初始化ShuffleC3"""
        super().__init__()
        
        隐藏通道 = output通道 // 2
        
        self.cv1 = 标准卷积(input通道, 隐藏通道, 1)
        self.cv2 = 标准卷积(input通道, 隐藏通道, 1)
        self.cv3 = 标准卷积(2 * 隐藏通道, output通道, 1)
        
        self.m = nn.Sequential(
            *[ShuffleNet单元(隐藏通道, 隐藏通道, 1) for _ in range(count)]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat([self.m(self.cv1(x)), self.cv2(x)], dim=1))
