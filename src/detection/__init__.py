#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
目标检测模块

提供目标检测的基础类、YOLOv5检测器和红外数据增强功能
"""

# 英文接口（推荐使用）
from .detector import BaseDetector, DetectionResult
from .yolov5_detector import YOLOv5Detector, create_yolov5_detector
from .data_augment import InfraredDataAugmentor, create_train_augmentor, create_val_augmentor

# 中文别名（向后兼容）
from .detector import 检测器基类, 检测结果
from .yolov5_detector import YOLOv5检测器, 创建YOLOv5检测器
from .data_augment import 红外数据增强器, 创建训练增强器, 创建验证增强器

__all__ = [
    # 英文接口
    'BaseDetector',
    'DetectionResult',
    'YOLOv5Detector',
    'create_yolov5_detector',
    'InfraredDataAugmentor',
    'create_train_augmentor',
    'create_val_augmentor',
    # 中文别名
    '检测器基类',
    '检测结果',
    'YOLOv5检测器',
    '创建YOLOv5检测器',
    '红外数据增强器',
    '创建训练增强器',
    '创建验证增强器',
]
