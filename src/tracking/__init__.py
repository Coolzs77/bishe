#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多目标跟踪模块
"""

from .tracker import BaseTracker, TrackObject, TrackingResult
from .deepsort_tracker import DeepSORTTracker, create_deepsort_tracker
from .bytetrack_tracker import ByteTrack, create_bytetrack_tracker
from .centertrack_tracker import (
    CenterTrack, create_centertrack_tracker,
    KalmanFilter, KalmanBoxTracker,
    xyxy_to_xywh, xywh_to_xyxy, xyxy_to_xyah, xyah_to_xyxy,
)

__all__ = [
    # 基类
    'BaseTracker',
    'TrackObject',
    'TrackingResult',
    # 跟踪器
    'DeepSORTTracker',
    'create_deepsort_tracker',
    'ByteTrack',
    'create_bytetrack_tracker',
    'CenterTrack',
    'create_centertrack_tracker',
    # CenterTrack 内部工具（保持向后兼容）
    'KalmanFilter',
    'KalmanBoxTracker',
    'xyxy_to_xywh',
    'xywh_to_xyxy',
    'xyxy_to_xyah',
    'xyah_to_xyxy',
]
