#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ByteTrack 跟踪器模块
使用官方 bytetracker 库实现
"""

import numpy as np
import torch
from typing import List, Optional

from .tracker import BaseTracker, TrackObject, TrackingResult
from bytetracker import BYTETracker


class ByteTrack(BaseTracker):
    """
    ByteTrack 多目标跟踪器

    封装官方 bytetracker 库，利用低置信度检测进行二次关联。

    Attributes:
        high_threshold: 高置信度阈值
        low_threshold: 低置信度阈值（由 bytetracker 内部处理）
        match_threshold: 匹配 IoU 阈值
        second_match_threshold: 二次匹配 IoU 阈值
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        high_threshold: float = 0.5,
        low_threshold: float = 0.1,
        match_threshold: float = 0.8,
        second_match_threshold: float = 0.5
    ):
        super().__init__(max_age, min_hits, iou_threshold)

        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.match_threshold = match_threshold
        self.second_match_threshold = second_match_threshold

        self._tracker = BYTETracker(
            track_thresh=high_threshold,
            track_buffer=max_age,
            match_thresh=match_threshold,
            frame_rate=30,
        )
        self.tracks: List = []
        self.lost_tracks: List = []

    def update(
        self,
        detections: np.ndarray,
        confidences: Optional[np.ndarray] = None,
        classes: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None
    ) -> TrackingResult:
        self.frame_count += 1

        if len(detections) == 0:
            detections = np.array([]).reshape(0, 4)

        if confidences is None:
            confidences = np.ones(len(detections))
        if classes is None:
            classes = np.zeros(len(detections), dtype=int)

        if len(detections) == 0:
            return TrackingResult(tracks=[], frame_id=self.frame_count)

        # Build (N, 6) tensor: [x1, y1, x2, y2, conf, cls]
        dets_np = np.concatenate([
            detections.astype(float),
            confidences.reshape(-1, 1).astype(float),
            classes.reshape(-1, 1).astype(float),
        ], axis=1)
        dets_tensor = torch.from_numpy(dets_np).float()

        outputs = self._tracker.update(dets_tensor, None)  # None = img_info not used by BYTETracker

        # Build a lookup from track_id → STrack for accessing actual track state
        track_lookup = {t.track_id: t for t in self._tracker.tracked_stracks}

        result_tracks = []
        if len(outputs) > 0:
            for row in outputs:
                x1, y1, x2, y2, track_id, cls, score = row
                tid = int(track_id)
                strack = track_lookup.get(tid)
                age = int(strack.frame_id - strack.start_frame + 1) if strack else 1
                hits = int(strack.tracklet_len) if strack else 1
                result_tracks.append(TrackObject(
                    track_id=tid,
                    bbox=np.array([x1, y1, x2, y2]),
                    confidence=float(score),
                    class_id=int(cls),
                    state='confirmed',
                    age=age,
                    hits=hits,
                    time_since_update=0,
                ))

        return TrackingResult(tracks=result_tracks, frame_id=self.frame_count)

    def reset(self) -> None:
        super().reset()
        self._tracker = BYTETracker(
            track_thresh=self.high_threshold,
            track_buffer=self.max_age,
            match_thresh=self.match_threshold,
            frame_rate=30,
        )
        self.tracks = []
        self.lost_tracks = []


def create_bytetrack_tracker(
    max_age: int = 30,
    min_hits: int = 3,
    iou_threshold: float = 0.3,
    high_threshold: float = 0.5,
    low_threshold: float = 0.1
) -> ByteTrack:
    return ByteTrack(
        max_age=max_age,
        min_hits=min_hits,
        iou_threshold=iou_threshold,
        high_threshold=high_threshold,
        low_threshold=low_threshold,
    )
