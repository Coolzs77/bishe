#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv5自定义模块
"""

from .attention import (
    SEAttention,
    ChannelAttention,
    SpatialAttention,
    CBAMAttention,
    CoordAttention,
    ECAAttention,
    get_attention_module,
)

__all__ = [
    'SE注意力',
    '通道注意力',
    '空间注意力',
    'CBAM注意力',
    '坐标注意力',
    'ECA注意力',
    '获取注意力模块',
]
