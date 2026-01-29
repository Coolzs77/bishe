#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation module providing detection and multi-object tracking metrics.
"""

from .detection_eval import DetectionEvaluator, evaluate_detection_model
from .tracking_eval import MOTEvaluator, evaluate_tracker, compare_trackers, print_comparison_table

__all__ = [
    # Detection evaluation
    'DetectionEvaluator',
    'evaluate_detection_model',
    # Tracking evaluation
    'MOTEvaluator',
    'evaluate_tracker',
    'compare_trackers',
    'print_comparison_table',
]
