#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSORT 跟踪器模块
使用官方 deep-sort-realtime 库实现
"""

import numpy as np

from .tracker import BaseTracker, TrackObject
from deep_sort_realtime.deepsort_tracker import DeepSort


class DeepSORTTracker(BaseTracker):
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3, max_cosine_distance=0.2, nn_budget=100, device='0'):
        super().__init__(max_age, min_hits, iou_threshold)
        self.n_init = min_hits
        self.max_cosine_distance = max_cosine_distance
        self.nn_budget = nn_budget

        use_gpu = device not in ('cpu', '-1')
        self._tracker = DeepSort(
            max_iou_distance=1.0 - iou_threshold,
            max_age=max_age,
            n_init=min_hits,
            max_cosine_distance=max_cosine_distance,
            nn_budget=nn_budget if nn_budget else None,
            embedder='mobilenet',
            half=False,
            bgr=True,
            embedder_gpu=use_gpu,
        )
        self.tracks = []

    def update(self, detections, confidences=None, ori_img=None, classes=None):
        self.frame_count += 1

        if len(detections) == 0:
            self.tracks = []
            return []

        if confidences is None:
            confidences = np.ones(len(detections))
        if classes is None:
            classes = np.zeros(len(detections), dtype=int)

        # Convert [x1, y1, x2, y2] → ([l, t, w, h], confidence, class_id)
        raw_detections = []
        for bbox, conf, cls in zip(detections, confidences, classes):
            x1, y1, x2, y2 = bbox
            ltwh = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            raw_detections.append((ltwh, float(conf), int(cls)))

        if ori_img is None:
            # ori_img is required by deep-sort-realtime for appearance embedding.
            # When it is absent (e.g. during unit tests), a blank frame is used.
            # In production, always pass the actual video frame so that the
            # MobileNet embedder produces valid appearance features.
            import warnings
            warnings.warn(
                "ori_img not provided to DeepSORTTracker.update(); "
                "a blank dummy frame will be used, which disables appearance features.",
                RuntimeWarning,
                stacklevel=2,
            )
            if raw_detections:
                max_x = int(max(d[0][0] + d[0][2] for d in raw_detections)) + 1
                max_y = int(max(d[0][1] + d[0][3] for d in raw_detections)) + 1
            else:
                max_x, max_y = 1, 1
            ori_img = np.zeros((max(max_y, 1), max(max_x, 1), 3), dtype=np.uint8)

        tracks = self._tracker.update_tracks(raw_detections, frame=ori_img)

        self.tracks = [t for t in tracks if not t.is_deleted()]

        outputs = []
        for track in self.tracks:
            if track.is_confirmed() or track.is_tentative():
                bbox = np.array(track.to_ltrb())
                conf = track.det_conf if track.det_conf is not None else 0.0
                cls = track.det_class if track.det_class is not None else 0
                state = 'confirmed' if track.is_confirmed() else 'tentative'
                outputs.append(TrackObject(
                    track_id=track.track_id,
                    bbox=bbox,
                    confidence=float(conf),
                    class_id=int(cls),
                    state=state,
                    age=track.age,
                    hits=track.hits,
                    time_since_update=track.time_since_update,
                ))
        return outputs

    def reset(self):
        super().reset()
        self._tracker.delete_all_tracks()
        self.tracks = []


def create_deepsort_tracker(max_age=30, min_hits=3, iou_threshold=0.3, max_cosine_distance=0.2, nn_budget=100):
    return DeepSORTTracker(max_age, min_hits, iou_threshold, max_cosine_distance, nn_budget)