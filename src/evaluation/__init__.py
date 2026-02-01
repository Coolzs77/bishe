#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate模块
提供目标检测和多目标跟踪的evaluate功能
"""

from .detection_eval import DetectionEvaluator, evaluate_detection_model
from .tracking_eval import MOTEvaluator, evaluate_tracker, compare_trackers, print_comparison_table

__all__ = [
    # 检测evaluate
    'DetectionEvaluator',
    'evaluate_detection_model',
    # 跟踪evaluate
    'MOTEvaluator',
    'evaluate_tracker',
    'compare_trackers',
    'print_comparison_table',
]
