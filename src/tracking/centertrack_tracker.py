#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CenterTracktracker模块

实现基于中心点的多目标跟踪算法
"""

import numpy as np
from typing import List, Optional, Tuple, Dict

from .tracker import BaseTracker, TrackObject, TrackingResult


# ---------------------------------------------------------------------------
# 卡尔曼滤波工具函数和类（内部实现，供 CenterTrack 使用）
# ---------------------------------------------------------------------------

def xyxy_to_xywh(bbox: np.ndarray) -> np.ndarray:
    """将 [x1,y1,x2,y2] 转换为 [cx,cy,w,h]"""
    bbox = np.asarray(bbox)
    if bbox.ndim == 1:
        x1, y1, x2, y2 = bbox
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
    result = np.zeros_like(bbox)
    result[:, 0] = (bbox[:, 0] + bbox[:, 2]) / 2
    result[:, 1] = (bbox[:, 1] + bbox[:, 3]) / 2
    result[:, 2] = bbox[:, 2] - bbox[:, 0]
    result[:, 3] = bbox[:, 3] - bbox[:, 1]
    return result


def xywh_to_xyxy(bbox: np.ndarray) -> np.ndarray:
    """将 [cx,cy,w,h] 转换为 [x1,y1,x2,y2]"""
    bbox = np.asarray(bbox)
    if bbox.ndim == 1:
        x_c, y_c, w, h = bbox
        return np.array([x_c - w / 2, y_c - h / 2, x_c + w / 2, y_c + h / 2])
    result = np.zeros_like(bbox)
    result[:, 0] = bbox[:, 0] - bbox[:, 2] / 2
    result[:, 1] = bbox[:, 1] - bbox[:, 3] / 2
    result[:, 2] = bbox[:, 0] + bbox[:, 2] / 2
    result[:, 3] = bbox[:, 1] + bbox[:, 3] / 2
    return result


def xyxy_to_xyah(bbox: np.ndarray) -> np.ndarray:
    """将 [x1,y1,x2,y2] 转换为 [cx,cy,aspect_ratio,height]"""
    bbox = np.asarray(bbox)
    if bbox.ndim == 1:
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2, w / max(h, 1e-6), h])
    result = np.zeros_like(bbox)
    w = bbox[:, 2] - bbox[:, 0]
    h = bbox[:, 3] - bbox[:, 1]
    result[:, 0] = (bbox[:, 0] + bbox[:, 2]) / 2
    result[:, 1] = (bbox[:, 1] + bbox[:, 3]) / 2
    result[:, 2] = w / np.maximum(h, 1e-6)
    result[:, 3] = h
    return result


def xyah_to_xyxy(bbox: np.ndarray) -> np.ndarray:
    """将 [cx,cy,aspect_ratio,height] 转换为 [x1,y1,x2,y2]"""
    bbox = np.asarray(bbox)
    if bbox.ndim == 1:
        x_c, y_c, a, h = bbox
        w = a * h
        return np.array([x_c - w / 2, y_c - h / 2, x_c + w / 2, y_c + h / 2])
    result = np.zeros_like(bbox)
    w = bbox[:, 2] * bbox[:, 3]
    h = bbox[:, 3]
    result[:, 0] = bbox[:, 0] - w / 2
    result[:, 1] = bbox[:, 1] - h / 2
    result[:, 2] = bbox[:, 0] + w / 2
    result[:, 3] = bbox[:, 1] + h / 2
    return result


class KalmanFilter:
    """
    卡尔曼滤波器（CenterTrack 内部使用）

    状态向量: [x, y, a, h, vx, vy, va, vh]
    """

    def __init__(self, std_weight_position: float = 1. / 20,
                 std_weight_velocity: float = 1. / 160):
        self.dim_x = 8
        self.dim_z = 4
        self._std_weight_position = std_weight_position
        self._std_weight_velocity = std_weight_velocity

        self._motion_mat = np.eye(self.dim_x)
        for i in range(4):
            self._motion_mat[i, 4 + i] = 1.0
        self._update_mat = np.eye(self.dim_z, self.dim_x)

    def initiate(self, measurement):
        mean = np.concatenate([measurement, np.zeros(4)])
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        return mean, np.diag(np.square(std))

    def predict(self, mean, covariance):
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(std_pos + std_vel))
        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean, covariance

    def project(self, mean, covariance):
        std = [self._std_weight_position * mean[3]] * 3 + [self._std_weight_position * mean[3]]
        std[2] = 1e-1
        innovation_cov = np.diag(np.square(std))
        mean = self._update_mat @ mean
        covariance = self._update_mat @ covariance @ self._update_mat.T + innovation_cov
        return mean, covariance

    def update(self, mean, covariance, measurement):
        projected_mean, projected_cov = self.project(mean, covariance)
        try:
            chol = np.linalg.cholesky(projected_cov)
            kalman_gain = np.linalg.solve(
                chol.T, np.linalg.solve(chol, self._update_mat @ covariance.T)
            ).T
        except np.linalg.LinAlgError:
            kalman_gain = covariance @ self._update_mat.T @ np.linalg.pinv(projected_cov)
        innovation = measurement - projected_mean
        new_mean = mean + kalman_gain @ innovation
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements, only_position=False):
        projected_mean, projected_cov = self.project(mean, covariance)
        if only_position:
            projected_mean = projected_mean[:2]
            projected_cov = projected_cov[:2, :2]
            measurements = measurements[:, :2]
        diff = measurements - projected_mean
        try:
            chol = np.linalg.cholesky(projected_cov)
            z = np.linalg.solve(chol, diff.T).T
            return np.sum(z ** 2, axis=1)
        except np.linalg.LinAlgError:
            return np.sum(diff @ np.linalg.pinv(projected_cov) * diff, axis=1)


class KalmanBoxTracker:
    """单目标卡尔曼滤波跟踪器（CenterTrack 内部使用）"""

    count = 0

    def __init__(self, bbox: np.ndarray, track_id: Optional[int] = None):
        self.kalman_filter = KalmanFilter()
        self.mean, self.covariance = self.kalman_filter.initiate(xyxy_to_xyah(bbox))
        if track_id is None:
            KalmanBoxTracker.count += 1
            self.track_id = KalmanBoxTracker.count
        else:
            self.track_id = track_id
        self.hits = 1
        self.time_since_update = 0
        self.age = 1
        self.history: List[np.ndarray] = []

    def predict(self) -> np.ndarray:
        if self.mean[6] + self.mean[2] <= 0:
            self.mean[6] = 0
        if self.mean[7] + self.mean[3] <= 0:
            self.mean[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1
        self.history.append(self.get_state())
        return self.get_state()

    def update(self, bbox: np.ndarray) -> None:
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, xyxy_to_xyah(bbox)
        )
        self.hits += 1
        self.time_since_update = 0
        self.history = []

    def get_state(self) -> np.ndarray:
        return xyah_to_xyxy(self.mean[:4])

    @classmethod
    def reset_count(cls):
        cls.count = 0


class CenterTrack(BaseTracker):
    """
    CenterTrack多目标tracker
    
    基于目标中心点位移的跟踪算法
    
    Attributes:
        center_threshold: 中心点距离阈值
        pre_thresh: 前一帧检测阈值
        tracks: 跟踪目标列表
        prev_centers: 上一帧的中心点
    """
    
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        center_threshold: float = 50.0,
        pre_thresh: float = 0.3
    ):
        """
        初始化CenterTracktracker
        
        Args:
            max_age: 目标最大存活帧数
            min_hits: 确认目标所需的最小命中次数
            iou_threshold: IoU匹配阈值
            center_threshold: 中心点距离阈值（像素）
            pre_thresh: 前一帧检测阈值
        """
        super().__init__(max_age, min_hits, iou_threshold)
        
        self.center_threshold = center_threshold
        self.pre_thresh = pre_thresh
        
        self.tracks: List[KalmanBoxTracker] = []
        self.prev_centers: Dict[int, np.ndarray] = {}
    
    def update(
        self,
        detections: np.ndarray,
        confidences: Optional[np.ndarray] = None,
        classes: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None,
        offsets: Optional[np.ndarray] = None
    ) -> TrackingResult:
        """
        更新tracker
        
        Args:
            detections: det_boxes数组，形状为 (N, 4)
            confidences: confidence数组
            classes: classes数组
            features: 特征数组（不使用）
            offsets: 中心点偏移量，形状为 (N, 2)
            
        Returns:
            跟踪results
        """
        self.frame_count += 1
        
        # 默认值处理
        if len(detections) == 0:
            detections = np.array([]).reshape(0, 4)
        if confidences is None:
            confidences = np.ones(len(detections))
        if classes is None:
            classes = np.zeros(len(detections), dtype=int)
        
        # 计算当前帧的中心点
        current_centers = self._compute_centers(detections)
        
        # 预测现有跟踪目标位置
        predicted_centers = {}
        for track in self.tracks:
            track.predict()
            predicted_box = track.get_state()
            predicted_centers[track.track_id] = self._compute_centers(predicted_box.reshape(1, -1))[0]
        
        # 如果提供了偏移量，使用偏移量推断前一帧位置
        if offsets is not None and len(offsets) > 0:
            prev_centers_from_offset = current_centers - offsets
        else:
            prev_centers_from_offset = None
        
        # 匹配检测和跟踪目标
        matched, unmatched_tracks, unmatched_dets = self._match_centers(
            detections, current_centers, predicted_centers, prev_centers_from_offset
        )
        
        # 更新匹配的跟踪目标
        for track_idx, det_idx in matched:
            self.tracks[track_idx].update(detections[det_idx])
            self.tracks[track_idx].class_id = int(classes[det_idx])
            self.tracks[track_idx].confidence = confidences[det_idx]
            self.prev_centers[self.tracks[track_idx].track_id] = current_centers[det_idx]
        
        # 处理未匹配的跟踪目标
        for track_idx in unmatched_tracks:
            track = self.tracks[track_idx]
            # 如果超时，删除；否则保留预测位置
            if track.time_since_update > self.max_age:
                if track.track_id in self.prev_centers:
                    del self.prev_centers[track.track_id]
        
        # 创建新跟踪目标
        for det_idx in unmatched_dets:
            if confidences[det_idx] >= self.pre_thresh:
                new_track = KalmanBoxTracker(detections[det_idx], self.get_next_id())
                new_track.class_id = int(classes[det_idx])
                new_track.confidence = confidences[det_idx]
                self.tracks.append(new_track)
                self.prev_centers[new_track.track_id] = current_centers[det_idx]
        
        # 删除超时的跟踪目标
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        
        # 生成results
        result_tracks = []
        for track in self.tracks:
            if track.hits >= self.min_hits or self.frame_count <= self.min_hits:
                track_obj = TrackObject(
                    track_id=track.track_id,
                    bbox=track.get_state(),
                    confidence=getattr(track, 'confidence', 1.0),
                    class_id=getattr(track, 'class_id', 0),
                    state='confirmed' if track.hits >= self.min_hits else 'tentative',
                    age=track.age,
                    hits=track.hits,
                    time_since_update=track.time_since_update
                )
                result_tracks.append(track_obj)
        
        return TrackingResult(tracks=result_tracks, frame_id=self.frame_count)
    
    def _compute_centers(self, boxes: np.ndarray) -> np.ndarray:
        """
        计算边界框的中心点
        
        Args:
            boxes: 边界框数组，形状为 (N, 4)
            
        Returns:
            中心点数组，形状为 (N, 2)
        """
        if len(boxes) == 0:
            return np.array([]).reshape(0, 2)
        
        centers = np.zeros((len(boxes), 2))
        centers[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2  # x_center
        centers[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2  # y_center
        return centers
    
    def _match_centers(
        self,
        detections: np.ndarray,
        current_centers: np.ndarray,
        predicted_centers: Dict[int, np.ndarray],
        prev_centers_from_offset: Optional[np.ndarray]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        基于中心点距离进行匹配
        
        Args:
            detections: det_boxes数组
            current_centers: 当前检测的中心点
            predicted_centers: 预测的跟踪目标中心点
            prev_centers_from_offset: 从偏移量推断的前一帧位置
            
        Returns:
            matched: 匹配对列表
            unmatched_tracks: 未匹配的跟踪索引
            unmatched_detections: 未匹配的检测索引
        """
        if len(self.tracks) == 0:
            return [], [], list(range(len(detections)))
        
        if len(detections) == 0:
            return [], list(range(len(self.tracks))), []
        
        # 计算距离矩阵
        n_tracks = len(self.tracks)
        n_dets = len(detections)
        
        cost_matrix = np.full((n_tracks, n_dets), self.center_threshold + 1)
        
        for i, track in enumerate(self.tracks):
            pred_center = predicted_centers.get(track.track_id)
            if pred_center is None:
                continue
            
            for j in range(n_dets):
                # 计算预测中心与当前检测中心的距离
                dist_pred = np.linalg.norm(pred_center - current_centers[j])
                
                # 如果有偏移量，也考虑偏移量推断的位置
                if prev_centers_from_offset is not None:
                    prev_center = self.prev_centers.get(track.track_id)
                    if prev_center is not None:
                        dist_offset = np.linalg.norm(prev_center - prev_centers_from_offset[j])
                        dist = min(dist_pred, dist_offset)
                    else:
                        dist = dist_pred
                else:
                    dist = dist_pred
                
                cost_matrix[i, j] = dist
        
        # 同时考虑IoU
        track_boxes = np.array([t.get_state() for t in self.tracks])
        iou_matrix = self.compute_iou_matrix(track_boxes, detections)
        
        # 组合代价：距离 + (1 - IoU) * scale
        iou_scale = self.center_threshold
        combined_cost = cost_matrix + (1 - iou_matrix) * iou_scale
        
        # 使用线性分配
        matched, unmatched_tracks, unmatched_dets = self.linear_assignment(
            combined_cost, self.center_threshold * 2
        )
        
        return matched, unmatched_tracks, unmatched_dets
    
    def reset(self) -> None:
        """重置tracker"""
        super().reset()
        self.tracks = []
        self.prev_centers = {}
        KalmanBoxTracker.reset_count()


def create_centertrack_tracker(
    max_age: int = 30,
    min_hits: int = 3,
    iou_threshold: float = 0.3,
    center_threshold: float = 50.0,
    pre_thresh: float = 0.3
) -> CenterTrack:
    """
    创建CenterTracktracker
    
    Args:
        max_age: 目标最大存活帧数
        min_hits: 确认所需的最小命中次数
        iou_threshold: IoU匹配阈值
        center_threshold: 中心点距离阈值
        pre_thresh: 前一帧检测阈值
        
    Returns:
        config好的CenterTracktracker
    """
    return CenterTrack(
        max_age=max_age,
        min_hits=min_hits,
        iou_threshold=iou_threshold,
        center_threshold=center_threshold,
        pre_thresh=pre_thresh
    )
