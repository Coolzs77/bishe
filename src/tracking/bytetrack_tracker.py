#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ByteTrack 跟踪器模块

使用 supervision 官方库中的 ByteTrack 实现，保持与 BaseTracker 接口兼容。
"""

import numpy as np
from typing import List, Optional

from .tracker import BaseTracker, TrackObject, TrackingResult

try:
    from supervision import ByteTrack as _ByteTrack, Detections as _Detections
    _BYTETRACK_AVAILABLE = True
except ImportError:
    _BYTETRACK_AVAILABLE = False


class ByteTrack(BaseTracker):
    """
    ByteTrack 多目标跟踪器（官方库包装）

    使用 supervision 库中的 ByteTrack 实现，保持与项目 BaseTracker 接口兼容。
    利用低置信度检测进行二次关联，提高跟踪召回率。

    Attributes:
        high_threshold: 高置信度阈值（用于首次关联）
        low_threshold: 低置信度阈值（仅在 update 中过滤）
        match_threshold: 匹配 IoU 阈值
        _tracker: 底层 supervision ByteTrack 实例
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 1,
        iou_threshold: float = 0.3,
        high_threshold: float = 0.5,
        low_threshold: float = 0.1,
        match_threshold: float = 0.8,
        second_match_threshold: float = 0.5,
    ):
        """
        初始化 ByteTrack 跟踪器

        Args:
            max_age: 目标最大存活帧数（lost_track_buffer）
            min_hits: 确认目标所需的最小命中次数（minimum_consecutive_frames）
            iou_threshold: IoU 匹配阈值（不直接使用，由 match_threshold 控制）
            high_threshold: 高置信度阈值（track_activation_threshold）
            low_threshold: 低置信度阈值（用于过滤噪声检测）
            match_threshold: 匹配阈值（minimum_matching_threshold）
            second_match_threshold: 二次匹配阈值（与 match_threshold 相同）
        """
        super().__init__(max_age, min_hits, iou_threshold)

        if not _BYTETRACK_AVAILABLE:
            raise ImportError(
                "supervision 未安装，请执行: pip install supervision"
            )

        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.match_threshold = match_threshold
        self.second_match_threshold = second_match_threshold

        self._tracker = _ByteTrack(
            track_activation_threshold=high_threshold,
            lost_track_buffer=max_age,
            minimum_matching_threshold=match_threshold,
            minimum_consecutive_frames=min_hits,
        )

    def update(
        self,
        detections: np.ndarray,
        confidences: Optional[np.ndarray] = None,
        classes: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None,
        ori_img: Optional[np.ndarray] = None,
    ) -> TrackingResult:
        """
        更新跟踪器

        Args:
            detections: 检测框数组，形状为 (N, 4)，格式 [x1, y1, x2, y2]
            confidences: 置信度数组，形状为 (N,)
            classes: 类别 ID 数组，形状为 (N,)
            features: 外观特征数组（ByteTrack 不使用）
            ori_img: 原始图像帧（ByteTrack 不使用）

        Returns:
            包含跟踪结果的 TrackingResult 对象
        """
        self.frame_count += 1

        detections = np.asarray(detections)
        if detections.ndim == 1:
            detections = detections.reshape(-1, 4)

        if len(detections) == 0:
            sv_dets = _Detections.empty()
            self._tracker.update_with_detections(sv_dets)
            return TrackingResult(tracks=[], frame_id=self.frame_count)

        if confidences is None:
            confidences = np.ones(len(detections))
        if classes is None:
            classes = np.zeros(len(detections), dtype=int)

        # 过滤低于最低阈值的检测
        valid_mask = confidences >= self.low_threshold
        detections = detections[valid_mask]
        confidences = confidences[valid_mask]
        classes = classes[valid_mask]

        if len(detections) == 0:
            sv_dets = _Detections.empty()
            self._tracker.update_with_detections(sv_dets)
            return TrackingResult(tracks=[], frame_id=self.frame_count)

        sv_dets = _Detections(
            xyxy=detections.astype(float),
            confidence=confidences.astype(float),
            class_id=classes.astype(int),
        )

        tracked = self._tracker.update_with_detections(sv_dets)

        result_tracks = []
        if tracked.tracker_id is not None:
            for i, tid in enumerate(tracked.tracker_id):
                bbox = tracked.xyxy[i]
                conf = float(tracked.confidence[i]) if tracked.confidence is not None else 1.0
                cls = int(tracked.class_id[i]) if tracked.class_id is not None else 0
                result_tracks.append(TrackObject(
                    track_id=int(tid),
                    bbox=np.array(bbox, dtype=float),
                    confidence=conf,
                    class_id=cls,
                    state='confirmed',
                    age=1,
                    hits=1,
                    time_since_update=0,
                ))

        return TrackingResult(tracks=result_tracks, frame_id=self.frame_count)

    def reset(self) -> None:
        """重置跟踪器状态"""
        super().reset()
        self._tracker.reset()


def create_bytetrack_tracker(
    max_age: int = 30,
    min_hits: int = 1,
    iou_threshold: float = 0.3,
    high_threshold: float = 0.5,
    low_threshold: float = 0.1,
    track_thresh: float = None,
    match_thresh: float = None,
) -> ByteTrack:
    """
    创建 ByteTrack 跟踪器

    Args:
        max_age: 目标最大存活帧数
        min_hits: 确认所需的最小命中次数
        iou_threshold: IoU 匹配阈值
        high_threshold: 高置信度阈值（也可通过 track_thresh 指定）
        low_threshold: 低置信度阈值
        track_thresh: high_threshold 的别名（优先使用）
        match_thresh: 匹配阈值的别名

    Returns:
        配置好的 ByteTrack 跟踪器
    """
    if track_thresh is not None:
        high_threshold = track_thresh
    match_threshold = match_thresh if match_thresh is not None else 0.8

    return ByteTrack(
        max_age=max_age,
        min_hits=min_hits,
        iou_threshold=iou_threshold,
        high_threshold=high_threshold,
        low_threshold=low_threshold,
        match_threshold=match_threshold,
    )
