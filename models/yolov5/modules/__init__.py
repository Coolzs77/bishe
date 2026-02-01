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
    'SEAttention',
    'ChannelAttention',
    'SpatialAttention',
    'CBAMAttention',
    'CoordAttention',
    'ECAAttention',
    'get_attention_module',
]
