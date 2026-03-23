#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CenterTrack 风格的统一跟踪器封装。"""

from typing import Optional

import numpy as np

from .tracker import TrackingResult
from .unified_tracker import UnifiedTracker


class CenterTrack(UnifiedTracker):
    """CenterTrack 封装。

    说明:
    - 保留 offsets 入参兼容旧调用
    - 采用中心/IoU稳定匹配，减少ID切换
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        center_threshold: float = 50.0,
        pre_thresh: float = 0.3,
        visible_lag: int = 10,
    ):
        del center_threshold
        super().__init__(
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold,
            high_threshold=pre_thresh,
            low_threshold=max(0.05, pre_thresh * 0.5),
            match_iou_threshold=max(0.25, iou_threshold),
            second_match_iou_threshold=0.15,
            reactivate_iou_threshold=0.2,
            use_low_score_match=True,
            class_aware=True,
            lost_track_buffer=max_age * 2,
            visible_lag=visible_lag,
        )

    def update(
        self,
        detections: np.ndarray,
        confidences: Optional[np.ndarray] = None,
        classes: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None,
        offsets: Optional[np.ndarray] = None,
    ) -> TrackingResult:
        del offsets, features
        return super().update(
            detections=detections,
            confidences=confidences,
            classes=classes,
            features=None,
        )


def create_centertrack_tracker(
    max_age: int = 30,
    min_hits: int = 3,
    iou_threshold: float = 0.3,
    center_threshold: float = 50.0,
    pre_thresh: float = 0.3,
    visible_lag: int = 10,
) -> CenterTrack:
    return CenterTrack(
        max_age=max_age,
        min_hits=min_hits,
        iou_threshold=iou_threshold,
        center_threshold=center_threshold,
        pre_thresh=pre_thresh,
        visible_lag=visible_lag,
    )
