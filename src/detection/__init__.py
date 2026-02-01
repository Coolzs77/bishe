#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
目标检测模块

提供目标检测的基础类、YOLOv5detector和红外data增强功能
"""

from .detector import BaseDetector, DetectionResult
from .yolov5_detector import YOLOv5Detector, create_yolov5_detector
from .data_augment import InfraredDataAugmentor, create_train_augmentor, create_val_augmentor

__all__ = [
    'BaseDetector',
    'DetectionResult',
    'YOLOv5Detector',
    'create_yolov5_detector',
    'InfraredDataAugmentor',
    'create_train_augmentor',
    'create_val_augmentor',
]
