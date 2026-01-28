#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ByteTrack跟踪器模块

实现ByteTrack多目标跟踪算法，利用低置信度检测进行跟踪
"""

import numpy as np
from typing import List, Optional, Tuple

from .tracker import BaseTracker, TrackObject, TrackingResult
from .kalman_filter import KalmanBoxTracker


class ByteTrack(BaseTracker):
    """
    ByteTrack多目标跟踪器
    
    利用低置信度检测框进行二次关联的跟踪算法
    
    Attributes:
        high_threshold: 高置信度阈值
        low_threshold: 低置信度阈值
        match_threshold: 匹配IoU阈值
        second_match_threshold: 二次匹配IoU阈值
        tracks: 活跃跟踪列表
        lost_tracks: 丢失跟踪列表
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
        """
        初始化ByteTrack跟踪器
        
        Args:
            max_age: 目标最大存活帧数
            min_hits: 确认目标所需的最小命中次数
            iou_threshold: IoU匹配阈值
            high_threshold: 高置信度阈值
            low_threshold: 低置信度阈值
            match_threshold: 第一次匹配的IoU阈值
            second_match_threshold: 第二次匹配的IoU阈值
        """
        super().__init__(max_age, min_hits, iou_threshold)
        
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.match_threshold = match_threshold
        self.second_match_threshold = second_match_threshold
        
        self.tracks: List[KalmanBoxTracker] = []
        self.lost_tracks: List[KalmanBoxTracker] = []
    
    def update(
        self,
        detections: np.ndarray,
        confidences: Optional[np.ndarray] = None,
        classes: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None
    ) -> TrackingResult:
        """
        更新跟踪器
        
        Args:
            detections: 检测框数组，形状为 (N, 4)
            confidences: 置信度数组
            classes: 类别数组
            features: 特征数组（ByteTrack不使用）
            
        Returns:
            跟踪结果
        """
        self.frame_count += 1
        
        # 默认值处理
        if len(detections) == 0:
            detections = np.array([]).reshape(0, 4)
        if confidences is None:
            confidences = np.ones(len(detections))
        if classes is None:
            classes = np.zeros(len(detections), dtype=int)
        
        # 分离高低置信度检测
        high_mask = confidences >= self.high_threshold
        low_mask = (confidences >= self.low_threshold) & (confidences < self.high_threshold)
        
        high_dets = detections[high_mask]
        high_confs = confidences[high_mask]
        high_classes = classes[high_mask]
        
        low_dets = detections[low_mask]
        low_confs = confidences[low_mask]
        low_classes = classes[low_mask]
        
        # 预测现有跟踪目标
        for track in self.tracks:
            track.predict()
        
        # 第一次关联：高置信度检测与活跃跟踪
        matched1, unmatched_tracks1, unmatched_dets1 = self._match(
            self.tracks, high_dets, self.match_threshold
        )
        
        # 更新匹配的跟踪目标
        for track_idx, det_idx in matched1:
            self.tracks[track_idx].update(high_dets[det_idx])
            self.tracks[track_idx].class_id = int(high_classes[det_idx])
            self.tracks[track_idx].conf = high_confs[det_idx]
        
        # 获取未匹配的跟踪目标
        remaining_tracks = [self.tracks[i] for i in unmatched_tracks1]
        
        # 第二次关联：低置信度检测与剩余跟踪目标
        if len(low_dets) > 0 and len(remaining_tracks) > 0:
            matched2, unmatched_tracks2, _ = self._match(
                remaining_tracks, low_dets, self.second_match_threshold
            )
            
            # 更新匹配的跟踪目标
            for rel_track_idx, det_idx in matched2:
                track = remaining_tracks[rel_track_idx]
                track.update(low_dets[det_idx])
                track.class_id = int(low_classes[det_idx])
                track.conf = low_confs[det_idx]
            
            # 更新未匹配的跟踪目标列表
            remaining_track_indices = [unmatched_tracks1[i] for i in unmatched_tracks2]
        else:
            remaining_track_indices = unmatched_tracks1
        
        # 处理丢失的跟踪目标
        for track_idx in remaining_track_indices:
            track = self.tracks[track_idx]
            if track.time_since_update < self.max_age:
                self.lost_tracks.append(track)
        
        # 第三次关联：高置信度未匹配检测与丢失跟踪
        unmatched_high_dets = [high_dets[i] for i in unmatched_dets1]
        unmatched_high_confs = [high_confs[i] for i in unmatched_dets1]
        unmatched_high_classes = [high_classes[i] for i in unmatched_dets1]
        
        if len(unmatched_high_dets) > 0 and len(self.lost_tracks) > 0:
            # 预测丢失跟踪的位置
            for track in self.lost_tracks:
                if track.time_since_update == 0:
                    track.predict()
            
            matched3, unmatched_lost, final_unmatched = self._match(
                self.lost_tracks, np.array(unmatched_high_dets), self.match_threshold
            )
            
            # 恢复匹配的丢失跟踪
            for lost_idx, det_idx in matched3:
                track = self.lost_tracks[lost_idx]
                track.update(unmatched_high_dets[det_idx])
                track.class_id = int(unmatched_high_classes[det_idx])
                track.conf = unmatched_high_confs[det_idx]
                self.tracks.append(track)
            
            # 更新丢失跟踪列表
            self.lost_tracks = [self.lost_tracks[i] for i in unmatched_lost]
            
            # 更新未匹配检测
            unmatched_high_dets = [unmatched_high_dets[i] for i in final_unmatched]
            unmatched_high_confs = [unmatched_high_confs[i] for i in final_unmatched]
            unmatched_high_classes = [unmatched_high_classes[i] for i in final_unmatched]
        
        # 为未匹配的高置信度检测创建新跟踪
        for det, conf, cls in zip(unmatched_high_dets, unmatched_high_confs, unmatched_high_classes):
            new_track = KalmanBoxTracker(det, self.get_next_id())
            new_track.class_id = int(cls)
            new_track.conf = conf
            self.tracks.append(new_track)
        
        # 删除超时的跟踪目标
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        self.lost_tracks = [t for t in self.lost_tracks if t.time_since_update <= self.max_age]
        
        # 生成结果
        result_tracks = []
        for track in self.tracks:
            if track.hits >= self.min_hits or self.frame_count <= self.min_hits:
                track_obj = TrackObject(
                    track_id=track.track_id,
                    bbox=track.get_state(),
                    confidence=getattr(track, 'conf', 1.0),
                    class_id=getattr(track, 'class_id', 0),
                    state='confirmed' if track.hits >= self.min_hits else 'tentative',
                    age=track.age,
                    hits=track.hits,
                    time_since_update=track.time_since_update
                )
                result_tracks.append(track_obj)
        
        return TrackingResult(tracks=result_tracks, frame_id=self.frame_count)
    
    def _match(
        self,
        tracks: List[KalmanBoxTracker],
        detections: np.ndarray,
        threshold: float
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        IoU匹配
        
        Args:
            tracks: 跟踪目标列表
            detections: 检测框数组
            threshold: 匹配阈值
            
        Returns:
            matched: 匹配对列表
            unmatched_tracks: 未匹配的跟踪索引
            unmatched_detections: 未匹配的检测索引
        """
        if len(tracks) == 0:
            return [], [], list(range(len(detections)))
        
        if len(detections) == 0:
            return [], list(range(len(tracks))), []
        
        # 获取跟踪目标的预测框
        track_boxes = np.array([t.get_state() for t in tracks])
        
        # 计算IoU矩阵
        iou_matrix = self.compute_iou_matrix(track_boxes, detections)
        
        # 转换为代价矩阵
        cost_matrix = 1 - iou_matrix
        
        # 使用线性分配
        matched, unmatched_tracks, unmatched_detections = self.linear_assignment(
            cost_matrix, threshold
        )
        
        return matched, unmatched_tracks, unmatched_detections
    
    def reset(self) -> None:
        """重置跟踪器"""
        super().reset()
        self.tracks = []
        self.lost_tracks = []
        KalmanBoxTracker.reset_count()


def create_bytetrack_tracker(
    max_age: int = 30,
    min_hits: int = 3,
    iou_threshold: float = 0.3,
    high_threshold: float = 0.5,
    low_threshold: float = 0.1
) -> ByteTrack:
    """
    创建ByteTrack跟踪器
    
    Args:
        max_age: 目标最大存活帧数
        min_hits: 确认所需的最小命中次数
        iou_threshold: IoU匹配阈值
        high_threshold: 高置信度阈值
        low_threshold: 低置信度阈值
        
    Returns:
        配置好的ByteTrack跟踪器
    """
    return ByteTrack(
        max_age=max_age,
        min_hits=min_hits,
        iou_threshold=iou_threshold,
        high_threshold=high_threshold,
        low_threshold=low_threshold
    )
