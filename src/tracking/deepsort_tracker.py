#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DeepSORT 风格的统一跟踪器封装。"""

from typing import Optional

import numpy as np

from .tracker import TrackingResult
from .unified_tracker import UnifiedTracker


class DeepSORTTracker(UnifiedTracker):
    """DeepSORT 封装。

    说明:
    - 保留原有调用签名(支持 ori_img / features 参数)
    - 当前默认使用几何 + 运动稳定策略，避免重依赖ReID导致的不稳定
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        max_cosine_distance: float = 0.2,
        nn_budget: int = 100,
        device: str = "0",
        visible_lag: int = 8,
    ):
        del max_cosine_distance, nn_budget, device
        super().__init__(
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold,
            high_threshold=0.25,
            low_threshold=0.05,
            match_iou_threshold=max(iou_threshold, 0.3),
            second_match_iou_threshold=0.2,
            reactivate_iou_threshold=0.25,
            use_low_score_match=True,
            class_aware=True,
            lost_track_buffer=max_age * 2,
            visible_lag=visible_lag,
        )

    def update(
        self,
        detections: np.ndarray,
        confidences: Optional[np.ndarray] = None,
        ori_img=None,
        classes: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None,
    ) -> TrackingResult:
        del ori_img, features
        return super().update(
            detections=detections,
            confidences=confidences,
            classes=classes,
            features=None,
        )


def create_deepsort_tracker(
    max_age: int = 30,
    min_hits: int = 3,
    iou_threshold: float = 0.3,
    max_cosine_distance: float = 0.2,
    nn_budget: int = 100,
    visible_lag: int = 8,
):
    return DeepSORTTracker(
        max_age=max_age,
        min_hits=min_hits,
        iou_threshold=iou_threshold,
        max_cosine_distance=max_cosine_distance,
        nn_budget=nn_budget,
        visible_lag=visible_lag,
    )
