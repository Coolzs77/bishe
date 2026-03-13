#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
注意力机制模块
包含SE、CBAM、CoordAttention等
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEAttention(nn.Module):
    """Squeeze-and-Excitation Attention"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ChannelAttention(nn.Module):
    """CBAM - Channel Attention"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """CBAM - Spatial Attention"""

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAMAttention(nn.Module):
    """CBAM: Convolutional Block Attention Module"""

    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class CoordAttention(nn.Module):
    """坐标注意力 - 强化热信号"""

    def __init__(self, channels, reduction=32):
        super().__init__()
        self.avg_pool_x = nn.AdaptiveAvgPool2d((None, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, None))

        self.conv_xu = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.BatchNorm2d(channels // reduction),
            nn.ReLU(inplace=True)
        )
        self.conv_xv = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.BatchNorm2d(channels // reduction),
            nn.ReLU(inplace=True)
        )
        self.conv_yu = nn.Sequential(
            nn.Conv2d(channels, channels, 1)
        )
        self.conv_yv = nn.Sequential(
            nn.Conv2d(channels, channels, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        x_avg = self.avg_pool_x(x)
        y_avg = self.avg_pool_y(x)

        xu = self.conv_xu(x_avg)
        xv = self.conv_xv(x_avg)
        yu = self.conv_yu(y_avg)
        yv = self.conv_yv(y_avg)

        bu = self.sigmoid(xu + yu)
        bv = self.sigmoid(xv + yv)

        return x * bu * bv


class ECAAttention(nn.Module):
    """Efficient Channel Attention"""

    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


def get_attention_module(attention_type, channels, **kwargs):
    """获取注意力模块"""
    attention_dict = {
        'se': SEAttention,
        'cbam': CBAMAttention,
        'coord': CoordAttention,
        'eca': ECAAttention,
    }

    if attention_type not in attention_dict:
        raise ValueError(f"Unknown attention type: {attention_type}")

    return attention_dict[attention_type](channels, **kwargs)