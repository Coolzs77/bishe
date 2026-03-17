#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轻量化模块 - 供models/yolov5/configs/*.yaml使用
"""

import torch
import torch.nn as nn
import math


class GhostModule(nn.Module):
    """Ghost Module: 通过线性变换生成幽灵特征"""

    def __init__(self, c1, c2, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super().__init__()
        self.out_channels = c2
        init_channels = math.ceil(c2 / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(c1, init_channels, kernel_size, stride,
                      kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1,
                      dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_channels, :, :]


class GhostBottleneck(nn.Module):
    """GhostBottleneck"""

    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        c_ = c2 // 2
        self.conv1 = GhostModule(c1, c_, 1, 2)
        self.conv2 = nn.Conv2d(c_, c_, k, s, k // 2, groups=c_, bias=False)
        self.bn2 = nn.BatchNorm2d(c_)
        self.conv3 = GhostModule(c_, c2, 1, 2, relu=False)
        self.shortcut = nn.Sequential() if c1 == c2 and s == 1 else None

    def forward(self, x):
        return self.conv3(self.bn2(torch.nn.functional.relu(self.conv2(self.conv1(x))))) + (
            x if self.shortcut is not None else 0)


class GhostC3(nn.Module):
    """GhostC3: 使用Ghost模块替代C3"""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, c_, 1, 1, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True),
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d(c_, c2, 1, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_, 3) for _ in range(n)))
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return self.cv2(self.m(self.cv1(x))) + (x if self.add else 0)