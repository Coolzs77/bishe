#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一多目标跟踪核心

目标:
- 统一不同跟踪器的行为，降低ID跳变
- 支持类别门控匹配，减少跨类误匹配
- 支持丢失轨迹重关联，降低目标时有时无
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .kalman_filter import KalmanBoxTracker
from .tracker import BaseTracker, TrackObject, TrackingResult


@dataclass
class _InternalTrack:
    """统一跟踪状态对象，封装卡尔曼轨迹与元数据。"""

    tracker: KalmanBoxTracker
    class_id: int
    confidence: float
    n_init: int
    ema_alpha: float = 0.7
    state: str = "tentative"
    class_votes: Dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.class_votes[self.class_id] = 1

    @property
    def track_id(self) -> int:
        return self.tracker.track_id

    @property
    def hits(self) -> int:
        return self.tracker.hits

    @property
    def age(self) -> int:
        return self.tracker.age

    @property
    def time_since_update(self) -> int:
        return self.tracker.time_since_update

    def predict(self) -> np.ndarray:
        return self.tracker.predict()

    def update(self, bbox: np.ndarray, class_id: int, confidence: float) -> None:
        self.tracker.update(bbox)

        # 使用投票稳定类别，避免类别在相邻帧抖动
        class_id = int(class_id)
        self.class_votes[class_id] = self.class_votes.get(class_id, 0) + 1
        self.class_id = max(self.class_votes.items(), key=lambda kv: kv[1])[0]

        # 使用EMA平滑置信度，避免标签显示抖动
        self.confidence = self.ema_alpha * float(confidence) + (1.0 - self.ema_alpha) * self.confidence

        if self.state == "tentative" and self.hits >= self.n_init:
            self.state = "confirmed"
        elif self.state == "lost":
            self.state = "confirmed"

    def mark_lost(self) -> None:
        self.state = "lost"

    def get_state(self) -> np.ndarray:
        return self.tracker.get_state()

    def to_track_object(self) -> TrackObject:
        return TrackObject(
            track_id=self.track_id,
            bbox=self.get_state(),
            confidence=float(self.confidence),
            class_id=int(self.class_id),
            state=self.state,
            age=int(self.age),
            hits=int(self.hits),
            time_since_update=int(self.time_since_update),
        )


class UnifiedTracker(BaseTracker):
    """稳定版统一跟踪器。"""

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        high_threshold: float = 0.5,
        low_threshold: float = 0.1,
        match_iou_threshold: float = 0.3,
        second_match_iou_threshold: float = 0.2,
        reactivate_iou_threshold: float = 0.2,
        use_low_score_match: bool = True,
        class_aware: bool = True,
        lost_track_buffer: Optional[int] = None,
        visible_lag: int = 1,
    ):
        super().__init__(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)

        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.match_iou_threshold = match_iou_threshold
        self.second_match_iou_threshold = second_match_iou_threshold
        self.reactivate_iou_threshold = reactivate_iou_threshold
        self.use_low_score_match = use_low_score_match
        self.class_aware = class_aware
        self.lost_track_buffer = int(lost_track_buffer if lost_track_buffer is not None else max_age * 2)
        self.visible_lag = max(0, int(visible_lag))

        self.active_tracks: List[_InternalTrack] = []
        self.lost_tracks: List[_InternalTrack] = []

    def update(
        self,
        detections: np.ndarray,
        confidences: Optional[np.ndarray] = None,
        classes: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None,
    ) -> TrackingResult:
        del features
        self.frame_count += 1

        detections = self._ensure_boxes(detections)
        confidences = self._ensure_confidences(confidences, len(detections))
        classes = self._ensure_classes(classes, len(detections))

        for track in self.active_tracks:
            track.predict()
        for track in self.lost_tracks:
            track.predict()

        hi_mask = confidences >= self.high_threshold
        lo_mask = (confidences >= self.low_threshold) & ~hi_mask

        hi_dets, hi_confs, hi_cls = detections[hi_mask], confidences[hi_mask], classes[hi_mask]
        lo_dets, lo_confs, lo_cls = detections[lo_mask], confidences[lo_mask], classes[lo_mask]

        matched_a, unmatched_active, unmatched_hi = self._associate(
            self.active_tracks,
            hi_dets,
            hi_cls,
            self.match_iou_threshold,
        )
        self._apply_matches(self.active_tracks, hi_dets, hi_cls, hi_confs, matched_a)

        if self.use_low_score_match and len(unmatched_active) > 0 and len(lo_dets) > 0:
            candidate_tracks = [self.active_tracks[i] for i in unmatched_active]
            matched_b_rel, unmatched_active_rel, _ = self._associate(
                candidate_tracks,
                lo_dets,
                lo_cls,
                self.second_match_iou_threshold,
            )
            matched_b = [(unmatched_active[t_idx], d_idx) for t_idx, d_idx in matched_b_rel]
            self._apply_matches(self.active_tracks, lo_dets, lo_cls, lo_confs, matched_b)
            unmatched_active = [unmatched_active[i] for i in unmatched_active_rel]

        remaining_hi_dets = hi_dets[unmatched_hi] if len(unmatched_hi) > 0 else np.zeros((0, 4), dtype=np.float32)
        remaining_hi_confs = hi_confs[unmatched_hi] if len(unmatched_hi) > 0 else np.zeros((0,), dtype=np.float32)
        remaining_hi_cls = hi_cls[unmatched_hi] if len(unmatched_hi) > 0 else np.zeros((0,), dtype=np.int32)

        if len(self.lost_tracks) > 0 and len(remaining_hi_dets) > 0:
            matched_lost, unmatched_lost, unmatched_rem = self._associate(
                self.lost_tracks,
                remaining_hi_dets,
                remaining_hi_cls,
                self.reactivate_iou_threshold,
            )
            self._apply_matches(self.lost_tracks, remaining_hi_dets, remaining_hi_cls, remaining_hi_confs, matched_lost)

            reactivated_indices = {i for i, _ in matched_lost}
            reactivated_tracks = [self.lost_tracks[i] for i in sorted(reactivated_indices)]
            for t in reactivated_tracks:
                t.state = "confirmed"
            self.active_tracks.extend(reactivated_tracks)
            self.lost_tracks = [self.lost_tracks[i] for i in unmatched_lost]

            remaining_hi_dets = remaining_hi_dets[unmatched_rem] if len(unmatched_rem) > 0 else np.zeros((0, 4), dtype=np.float32)
            remaining_hi_confs = remaining_hi_confs[unmatched_rem] if len(unmatched_rem) > 0 else np.zeros((0,), dtype=np.float32)
            remaining_hi_cls = remaining_hi_cls[unmatched_rem] if len(unmatched_rem) > 0 else np.zeros((0,), dtype=np.int32)

        self._handle_unmatched_active(unmatched_active)

        for det, conf, cls in zip(remaining_hi_dets, remaining_hi_confs, remaining_hi_cls):
            self._create_new_track(det, float(conf), int(cls))

        self._prune_lost_tracks()

        outputs: List[TrackObject] = []
        for track in self.active_tracks:
            if track.state == "confirmed" and track.time_since_update <= self.visible_lag:
                outputs.append(track.to_track_object())
            elif self.frame_count <= self.min_hits and track.time_since_update == 0:
                outputs.append(track.to_track_object())

        return TrackingResult(tracks=outputs, frame_id=self.frame_count)

    def reset(self) -> None:
        super().reset()
        self.active_tracks = []
        self.lost_tracks = []
        KalmanBoxTracker.reset_count()

    def _create_new_track(self, bbox: np.ndarray, confidence: float, class_id: int) -> None:
        kf_track = KalmanBoxTracker(bbox, self.get_next_id())
        track = _InternalTrack(
            tracker=kf_track,
            class_id=class_id,
            confidence=confidence,
            n_init=self.min_hits,
            state="tentative",
        )
        self.active_tracks.append(track)

    def _handle_unmatched_active(self, unmatched_indices: List[int]) -> None:
        if not unmatched_indices:
            return

        keep_active: List[_InternalTrack] = []
        move_lost: List[_InternalTrack] = []

        unmatched_set = set(unmatched_indices)
        for idx, track in enumerate(self.active_tracks):
            if idx in unmatched_set and track.time_since_update > self.max_age:
                track.mark_lost()
                move_lost.append(track)
            else:
                keep_active.append(track)

        self.active_tracks = keep_active
        self.lost_tracks.extend(move_lost)

    def _prune_lost_tracks(self) -> None:
        self.lost_tracks = [t for t in self.lost_tracks if t.time_since_update <= self.lost_track_buffer]

    def _apply_matches(
        self,
        tracks: List[_InternalTrack],
        dets: np.ndarray,
        classes: np.ndarray,
        confs: np.ndarray,
        matched: List[Tuple[int, int]],
    ) -> None:
        for t_idx, d_idx in matched:
            tracks[t_idx].update(dets[d_idx], int(classes[d_idx]), float(confs[d_idx]))

    def _associate(
        self,
        tracks: List[_InternalTrack],
        detections: np.ndarray,
        det_classes: np.ndarray,
        iou_thresh: float,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if len(tracks) == 0:
            return [], [], list(range(len(detections)))
        if len(detections) == 0:
            return [], list(range(len(tracks))), []

        track_boxes = np.array([t.get_state() for t in tracks], dtype=np.float32)
        iou = self.compute_iou_matrix(track_boxes, detections)
        if np.ndim(iou) == 0:
            iou = np.array([[float(iou)]], dtype=np.float32)
        elif iou.ndim == 1:
            iou = iou.reshape(len(tracks), len(detections))

        if self.class_aware:
            for i, track in enumerate(tracks):
                cls_mask = det_classes == int(track.class_id)
                iou[i, ~cls_mask] = 0.0

        cost = 1.0 - iou
        matched, unmatched_tracks, unmatched_dets = self.linear_assignment(cost, threshold=1.0 - float(iou_thresh))
        return matched, unmatched_tracks, unmatched_dets

    @staticmethod
    def _ensure_boxes(detections: np.ndarray) -> np.ndarray:
        if detections is None:
            return np.zeros((0, 4), dtype=np.float32)
        arr = np.asarray(detections, dtype=np.float32)
        if arr.size == 0:
            return np.zeros((0, 4), dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 4)
        return arr

    @staticmethod
    def _ensure_confidences(confidences: Optional[np.ndarray], n: int) -> np.ndarray:
        if confidences is None:
            return np.ones((n,), dtype=np.float32)
        arr = np.asarray(confidences, dtype=np.float32).reshape(-1)
        if len(arr) < n:
            padded = np.ones((n,), dtype=np.float32)
            padded[: len(arr)] = arr
            return padded
        return arr[:n]

    @staticmethod
    def _ensure_classes(classes: Optional[np.ndarray], n: int) -> np.ndarray:
        if classes is None:
            return np.zeros((n,), dtype=np.int32)
        arr = np.asarray(classes, dtype=np.int32).reshape(-1)
        if len(arr) < n:
            padded = np.zeros((n,), dtype=np.int32)
            padded[: len(arr)] = arr
            return padded
        return arr[:n]
