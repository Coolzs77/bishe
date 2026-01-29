#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSORTtracker模块

实现DeepSORT多目标跟踪算法，结合外观特征和运动model
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any

from .tracker import BaseTracker, TrackObject, TrackingResult
from .kalman_filter import KalmanFilter, xyxy_to_xyah, xyah_to_xyxy


class DeepSORTTrack:
    """
    DeepSORT跟踪目标
    
    封装单个目标的状态和特征
    
    Attributes:
        track_id: 跟踪ID
        mean: 状态均值
        covariance: 状态协方差
        features: 外观特征历史
        class_id: classesID
        confidence: confidence
        state: 跟踪状态
        hits: 命中次数
        age: 目标存在帧数
        time_since_update: 自上次更新以来的帧数
    """
    
    def __init__(
        self,
        track_id: int,
        bbox: np.ndarray,
        feature: Optional[np.ndarray] = None,
        class_id: int = 0,
        confidence: float = 1.0,
        n_init: int = 3,
        max_features: int = 100
    ):
        """
        初始化DeepSORT跟踪目标
        
        Args:
            track_id: 跟踪ID
            bbox: 初始边界框 [x1, y1, x2, y2]
            feature: 初始外观特征
            class_id: classesID
            confidence: confidence
            n_init: 确认所需的命中次数
            max_features: 最大保存特征数
        """
        self.track_id = track_id
        self.class_id = class_id
        self.confidence = confidence
        
        # 卡尔曼滤波器
        self.kf = KalmanFilter()
        measurement = xyxy_to_xyah(bbox)
        self.mean, self.covariance = self.kf.initiate(measurement)
        
        # 外观特征
        self.features: List[np.ndarray] = []
        self.max_features = max_features
        if feature is not None:
            self.features.append(feature)
        
        # 跟踪状态
        self.state = 'tentative'
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.n_init = n_init
    
    def predict(self) -> None:
        """预测下一帧状态"""
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1
    
    def update(
        self,
        bbox: np.ndarray,
        feature: Optional[np.ndarray] = None,
        confidence: float = 1.0
    ) -> None:
        """
        使用观测更新状态
        
        Args:
            bbox: 观测的边界框
            feature: 外观特征
            confidence: confidence
        """
        measurement = xyxy_to_xyah(bbox)
        self.mean, self.covariance = self.kf.update(self.mean, self.covariance, measurement)
        
        # 更新特征
        if feature is not None:
            self.features.append(feature)
            if len(self.features) > self.max_features:
                self.features.pop(0)
        
        self.confidence = confidence
        self.hits += 1
        self.time_since_update = 0
        
        # 更新状态
        if self.state == 'tentative' and self.hits >= self.n_init:
            self.state = 'confirmed'
    
    def mark_missed(self) -> None:
        """标记目标丢失"""
        if self.state == 'tentative':
            self.state = 'deleted'
    
    def is_deleted(self) -> bool:
        """检查是否已删除"""
        return self.state == 'deleted'
    
    def is_confirmed(self) -> bool:
        """检查是否已确认"""
        return self.state == 'confirmed'
    
    def get_bbox(self) -> np.ndarray:
        """获取当前边界框"""
        return xyah_to_xyxy(self.mean[:4])
    
    def to_track_object(self) -> TrackObject:
        """convert为TrackObject"""
        return TrackObject(
            track_id=self.track_id,
            bbox=self.get_bbox(),
            confidence=self.confidence,
            class_id=self.class_id,
            state=self.state,
            age=self.age,
            hits=self.hits,
            time_since_update=self.time_since_update,
            features=self.features[-1] if self.features else None
        )


class DeepSORTTracker(BaseTracker):
    """
    DeepSORT多目标tracker
    
    结合外观特征和卡尔曼滤波的多目标跟踪算法
    
    Attributes:
        max_cosine_distance: 最大余弦距离
        nn_budget: 特征库容量
        max_iou_distance: 最大IoU距离
        n_init: 确认所需的命中次数
        tracks: 跟踪目标列表
    """
    
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        max_cosine_distance: float = 0.3,
        nn_budget: int = 100,
        max_iou_distance: float = 0.7
    ):
        """
        初始化DeepSORTtracker
        
        Args:
            max_age: 目标最大存活帧数
            min_hits: 确认目标所需的最小命中次数
            iou_threshold: IoU匹配阈值
            max_cosine_distance: 最大余弦距离
            nn_budget: 特征库容量
            max_iou_distance: IoU级联匹配的最大距离
        """
        super().__init__(max_age, min_hits, iou_threshold)
        
        self.max_cosine_distance = max_cosine_distance
        self.nn_budget = nn_budget
        self.max_iou_distance = max_iou_distance
        self.n_init = min_hits
        
        self.tracks: List[DeepSORTTrack] = []
    
    def update(
        self,
        detections: np.ndarray,
        confidences: Optional[np.ndarray] = None,
        classes: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None
    ) -> TrackingResult:
        """
        更新tracker
        
        Args:
            detections: det_boxes数组，形状为 (N, 4)
            confidences: confidence数组
            classes: classes数组
            features: 特征数组
            
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
        
        # 预测步骤
        for track in self.tracks:
            track.predict()
        
        # 匹配步骤
        if features is not None and len(features) > 0:
            # 使用外观特征进行级联匹配
            matched, unmatched_tracks, unmatched_detections = self._cascade_matching(
                detections, features, confidences, classes
            )
        else:
            # 仅使用IoU匹配
            matched, unmatched_tracks, unmatched_detections = self._iou_matching(
                detections, confidences, classes
            )
        
        # 更新匹配的跟踪目标
        for track_idx, det_idx in matched:
            feature = features[det_idx] if features is not None else None
            self.tracks[track_idx].update(
                detections[det_idx],
                feature=feature,
                confidence=confidences[det_idx]
            )
        
        # 处理未匹配的跟踪目标
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        
        # 创建新跟踪目标
        for det_idx in unmatched_detections:
            feature = features[det_idx] if features is not None else None
            new_track = DeepSORTTrack(
                track_id=self.get_next_id(),
                bbox=detections[det_idx],
                feature=feature,
                class_id=int(classes[det_idx]),
                confidence=confidences[det_idx],
                n_init=self.n_init
            )
            self.tracks.append(new_track)
        
        # 删除丢失的跟踪目标
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        
        # 删除超时的跟踪目标
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        
        # 生成results
        result_tracks = [t.to_track_object() for t in self.tracks if t.is_confirmed()]
        
        return TrackingResult(tracks=result_tracks, frame_id=self.frame_count)
    
    def _cascade_matching(
        self,
        detections: np.ndarray,
        features: np.ndarray,
        confidences: np.ndarray,
        classes: np.ndarray
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        级联匹配
        
        首先使用外观特征匹配已确认的目标，然后使用IoU匹配剩余目标
        
        Returns:
            matched: 匹配对列表
            unmatched_tracks: 未匹配的跟踪索引
            unmatched_detections: 未匹配的检测索引
        """
        # 分离确认和未确认的跟踪目标
        confirmed_indices = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_indices = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]
        
        # 级联匹配确认的目标（按time_since_update分组）
        matched = []
        unmatched_detections = list(range(len(detections)))
        
        for age in range(self.max_age):
            if len(unmatched_detections) == 0:
                break
            
            # 找到time_since_update等于age的确认目标
            track_indices = [
                i for i in confirmed_indices 
                if self.tracks[i].time_since_update == age
            ]
            
            if len(track_indices) == 0:
                continue
            
            # 计算外观距离
            cost_matrix = self._appearance_cost(track_indices, unmatched_detections, features)
            
            # 匹配
            matches, _, unmatched_dets = self.linear_assignment(
                cost_matrix, self.max_cosine_distance
            )
            
            # 更新results
            for m in matches:
                matched.append((track_indices[m[0]], unmatched_detections[m[1]]))
            
            unmatched_detections = [unmatched_detections[i] for i in unmatched_dets]
        
        # 未匹配的确认目标
        matched_tracks = set(m[0] for m in matched)
        unmatched_confirmed = [i for i in confirmed_indices if i not in matched_tracks]
        
        # IoU匹配剩余的检测和跟踪目标
        candidate_tracks = unconfirmed_indices + unmatched_confirmed
        
        if len(candidate_tracks) > 0 and len(unmatched_detections) > 0:
            # 计算IoU代价矩阵
            track_boxes = np.array([self.tracks[i].get_bbox() for i in candidate_tracks])
            det_boxes = detections[unmatched_detections]
            
            iou_matrix = self.compute_iou_matrix(track_boxes, det_boxes)
            cost_matrix = 1 - iou_matrix
            
            # 匹配
            iou_matches, unmatched_t, unmatched_d = self.linear_assignment(
                cost_matrix, self.max_iou_distance
            )
            
            # 更新results
            for m in iou_matches:
                matched.append((candidate_tracks[m[0]], unmatched_detections[m[1]]))
            
            unmatched_tracks = [candidate_tracks[i] for i in unmatched_t]
            unmatched_detections = [unmatched_detections[i] for i in unmatched_d]
        else:
            unmatched_tracks = candidate_tracks
        
        return matched, unmatched_tracks, unmatched_detections
    
    def _iou_matching(
        self,
        detections: np.ndarray,
        confidences: np.ndarray,
        classes: np.ndarray
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        仅使用IoU进行匹配
        
        Returns:
            matched: 匹配对列表
            unmatched_tracks: 未匹配的跟踪索引
            unmatched_detections: 未匹配的检测索引
        """
        if len(self.tracks) == 0:
            return [], [], list(range(len(detections)))
        
        if len(detections) == 0:
            return [], list(range(len(self.tracks))), []
        
        # 计算IoU代价矩阵
        track_boxes = np.array([t.get_bbox() for t in self.tracks])
        iou_matrix = self.compute_iou_matrix(track_boxes, detections)
        cost_matrix = 1 - iou_matrix
        
        # 匹配
        matched, unmatched_tracks, unmatched_detections = self.linear_assignment(
            cost_matrix, self.max_iou_distance
        )
        
        return matched, unmatched_tracks, unmatched_detections
    
    def _appearance_cost(
        self,
        track_indices: List[int],
        detection_indices: List[int],
        features: np.ndarray
    ) -> np.ndarray:
        """
        计算外观代价矩阵
        
        Args:
            track_indices: 跟踪目标索引
            detection_indices: 检测索引
            features: 检测特征
            
        Returns:
            代价矩阵
        """
        cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
        
        for i, track_idx in enumerate(track_indices):
            track = self.tracks[track_idx]
            
            if len(track.features) == 0:
                cost_matrix[i, :] = self.max_cosine_distance
                continue
            
            # 计算与所有检测的余弦距离
            track_features = np.array(track.features)
            
            for j, det_idx in enumerate(detection_indices):
                det_feature = features[det_idx]
                
                # 计算最小余弦距离
                distances = self._cosine_distance(track_features, det_feature)
                cost_matrix[i, j] = np.min(distances)
        
        return cost_matrix
    
    @staticmethod
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        计算余弦距离
        
        Args:
            a: 特征数组，形状为 (N, D) 或 (D,)
            b: 特征向量，形状为 (D,)
            
        Returns:
            余弦距离
        """
        a = np.asarray(a)
        b = np.asarray(b)
        
        if a.ndim == 1:
            a = a.reshape(1, -1)
        
        # 归一化
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-6)
        b_norm = b / (np.linalg.norm(b) + 1e-6)
        
        # 余弦相似度
        cosine_sim = a_norm @ b_norm
        
        # 余弦距离
        return 1 - cosine_sim
    
    def reset(self) -> None:
        """重置tracker"""
        super().reset()
        self.tracks = []


def create_deepsort_tracker(
    max_age: int = 30,
    min_hits: int = 3,
    iou_threshold: float = 0.3,
    max_cosine_distance: float = 0.3,
    nn_budget: int = 100
) -> DeepSORTTracker:
    """
    创建DeepSORTtracker
    
    Args:
        max_age: 目标最大存活帧数
        min_hits: 确认所需的最小命中次数
        iou_threshold: IoU匹配阈值
        max_cosine_distance: 最大余弦距离
        nn_budget: 特征库容量
        
    Returns:
        config好的DeepSORTtracker
    """
    return DeepSORTTracker(
        max_age=max_age,
        min_hits=min_hits,
        iou_threshold=iou_threshold,
        max_cosine_distance=max_cosine_distance,
        nn_budget=nn_budget
    )
