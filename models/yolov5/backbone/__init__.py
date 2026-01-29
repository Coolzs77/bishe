#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轻量化骨干网络模块
"""

from .lightweight import (
    标准卷积,
    深度可分离卷积,
    Ghost模块,
    Ghost瓶颈,
    Shuffle通道,
    ShuffleNet单元,
    GhostC3,
    ShuffleC3,
)

__all__ = [
    '标准卷积',
    '深度可分离卷积',
    'Ghost模块',
    'Ghost瓶颈',
    'Shuffle通道',
    'ShuffleNet单元',
    'GhostC3',
    'ShuffleC3',
]
