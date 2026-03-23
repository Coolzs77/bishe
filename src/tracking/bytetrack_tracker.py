#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ByteTrack 风格的统一跟踪器封装。"""

from .unified_tracker import UnifiedTracker


class ByteTrack(UnifiedTracker):
    """ByteTrack 封装: 双阈值 + 低分框二次关联。"""

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        high_threshold: float = 0.5,
        low_threshold: float = 0.1,
        match_threshold: float = 0.3,
        second_match_threshold: float = 0.2,
        visible_lag: int = 8,
    ):
        super().__init__(
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold,
            high_threshold=high_threshold,
            low_threshold=low_threshold,
            match_iou_threshold=match_threshold,
            second_match_iou_threshold=second_match_threshold,
            reactivate_iou_threshold=match_threshold,
            use_low_score_match=True,
            class_aware=True,
            lost_track_buffer=max_age * 2,
            visible_lag=visible_lag,
        )


def create_bytetrack_tracker(
    max_age: int = 30,
    min_hits: int = 3,
    iou_threshold: float = 0.3,
    high_threshold: float = 0.5,
    low_threshold: float = 0.1,
    visible_lag: int = 8,
) -> ByteTrack:
    """
    创建ByteTracktracker
    
    Args:
        max_age: 目标最大存活帧数
        min_hits: 确认所需的最小命中次数
        iou_threshold: IoU匹配阈值
        high_threshold: 高confidence阈值
        low_threshold: 低confidence阈值
        
    Returns:
        config好的ByteTracktracker
    """
    return ByteTrack(
        max_age=max_age,
        min_hits=min_hits,
        iou_threshold=iou_threshold,
        high_threshold=high_threshold,
        low_threshold=low_threshold,
        visible_lag=visible_lag,
    )
