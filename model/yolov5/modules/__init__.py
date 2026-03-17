#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv5自定义模块 - 导出所有模块供YAML使用
"""

from .attention import CoordAttention, SEAttention, CBAMAttention
from .lightweight import GhostC3, GhostModule

__all__ = [
    'CoordAttention',
    'SEAttention',
    'CBAMAttention',
    'GhostC3',
    'GhostModule',
]