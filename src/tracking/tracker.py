#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tracker基类模块

提供多目标跟踪的基础抽象类和跟踪目标data类
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import numpy as np


@dataclass
class TrackObject:
    """
    跟踪目标data类
    
    存储单个跟踪目标的状态信息
    
    Attributes:
        track_id: 跟踪ID
        bbox: 边界框 [x1, y1, x2, y2]
        confidence: confidence
        class_id: classesID
        state: 跟踪状态 ('tentative', 'confirmed', 'lost')
        age: 目标存在帧数
        hits: 连续命中次数
        time_since_update: 自上次更新以来的帧数
        features: 外观特征向量
    """
    track_id: int
    bbox: np.ndarray
    confidence: float = 1.0
    class_id: int = 0
    state: str = 'tentative'
    age: int = 1
    hits: int = 1
    time_since_update: int = 0
    features: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """初始化postprocess"""
        if not isinstance(self.bbox, np.ndarray):
            self.bbox = np.array(self.bbox)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        convert为字典格式
        
        Returns:
            包含目标信息的字典
        """
        return {
            'track_id': self.track_id,
            'bbox': self.bbox.tolist(),
            'confidence': self.confidence,
            'class_id': self.class_id,
            'state': self.state,
            'age': self.age,
            'hits': self.hits,
            'time_since_update': self.time_since_update
        }


@dataclass
class TrackingResult:
    """
    跟踪resultsdata类
    
    存储一帧的所有跟踪results
    
    Attributes:
        tracks: 跟踪目标列表
        frame_id: 帧编号
    """
    tracks: List[TrackObject] = field(default_factory=list)
    frame_id: int = 0
    
    def __len__(self) -> int:
        """返回跟踪目标count"""
        return len(self.tracks)
    
    def __getitem__(self, idx: int) -> TrackObject:
        """获取指定索引的跟踪目标"""
        return self.tracks[idx]
    
    def get_boxes(self) -> np.ndarray:
        """
        获取所有边界框
        
        Returns:
            边界框数组，形状为 (N, 4)
        """
        if len(self.tracks) == 0:
            return np.array([]).reshape(0, 4)
        return np.array([t.bbox for t in self.tracks])
    
    def get_ids(self) -> np.ndarray:
        """
        获取所有跟踪ID
        
        Returns:
            跟踪ID数组
        """
        return np.array([t.track_id for t in self.tracks])
    
    def get_confidences(self) -> np.ndarray:
        """
        获取所有confidence
        
        Returns:
            confidence数组
        """
        return np.array([t.confidence for t in self.tracks])
    
    def get_confirmed_tracks(self) -> 'TrackingResult':
        """
        获取已确认的跟踪目标
        
        Returns:
            只包含已确认目标的跟踪results
        """
        confirmed = [t for t in self.tracks if t.state == 'confirmed']
        return TrackingResult(tracks=confirmed, frame_id=self.frame_id)
    
    def to_list(self) -> List[Dict[str, Any]]:
        """
        convert为字典列表格式
        
        Returns:
            包含所有跟踪目标信息的字典列表
        """
        return [t.to_dict() for t in self.tracks]


class BaseTracker(ABC):
    """
    tracker基类
    
    提供多目标tracker的抽象接口
    
    Attributes:
        max_age: 目标最大存活帧数
        min_hits: 确认目标所需的最小命中次数
        iou_threshold: IoU匹配阈值
        frame_count: 帧计数器
        tracks: 当前跟踪列表
        next_id: 下一个可用的跟踪ID
    """
    
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3
    ):
        """
        初始化tracker
        
        Args:
            max_age: 目标最大存活帧数，超过此帧数未更新的目标将被删除
            min_hits: 确认目标所需的最小命中次数
            iou_threshold: IoU匹配阈值
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        self.frame_count = 0
        self.tracks: List[TrackObject] = []
        self.next_id = 1
    
    @abstractmethod
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
            detections: det_boxes数组，形状为 (N, 4)，格式为 [x1, y1, x2, y2]
            confidences: confidence数组，形状为 (N,)
            classes: classes数组，形状为 (N,)
            features: 特征数组，形状为 (N, D)
            
        Returns:
            跟踪results
        """
        pass
    
    def reset(self) -> None:
        """重置tracker状态"""
        self.frame_count = 0
        self.tracks = []
        self.next_id = 1
    
    def get_next_id(self) -> int:
        """
        获取下一个跟踪ID
        
        Returns:
            新的跟踪ID
        """
        track_id = self.next_id
        self.next_id += 1
        return track_id
    
    @staticmethod
    def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """
        计算两个边界框的IoU
        
        Args:
            box1: 第一个边界框 [x1, y1, x2, y2]
            box2: 第二个边界框 [x1, y1, x2, y2]
            
        Returns:
            IoU值
        """
        # 计算交集
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        # 计算各自面积
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # 计算IoU
        union_area = area1 + area2 - inter_area
        if union_area <= 0:
            return 0.0
        
        return inter_area / union_area
    
    @staticmethod
    def compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """
        计算两组边界框的IoU矩阵
        
        Args:
            boxes1: 第一组边界框，形状为 (N, 4)
            boxes2: 第二组边界框，形状为 (M, 4)
            
        Returns:
            IoU矩阵，形状为 (N, M)
        """
        if len(boxes1) == 0 or len(boxes2) == 0:
            return np.zeros((len(boxes1), len(boxes2)))
        
        # 扩展维度以便广播
        boxes1 = boxes1[:, np.newaxis, :]  # (N, 1, 4)
        boxes2 = boxes2[np.newaxis, :, :]  # (1, M, 4)
        
        # 计算交集
        x1 = np.maximum(boxes1[..., 0], boxes2[..., 0])
        y1 = np.maximum(boxes1[..., 1], boxes2[..., 1])
        x2 = np.minimum(boxes1[..., 2], boxes2[..., 2])
        y2 = np.minimum(boxes1[..., 3], boxes2[..., 3])
        
        inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # 计算各自面积
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        
        # 计算IoU
        union_area = area1 + area2 - inter_area
        iou_matrix = np.where(union_area > 0, inter_area / union_area, 0)
        
        return iou_matrix.squeeze()
    
    @staticmethod
    def linear_assignment(cost_matrix: np.ndarray, threshold: float = 0.5) -> Tuple[List, List, List]:
        """
        线性分配（匈牙利算法）
        
        Args:
            cost_matrix: 代价矩阵，形状为 (N, M)
            threshold: 匹配阈值
            
        Returns:
            matched_indices: 匹配的索引对列表
            unmatched_rows: 未匹配的行索引列表
            unmatched_cols: 未匹配的列索引列表
        """
        if cost_matrix.size == 0:
            return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
        
        try:
            from scipy.optimize import linear_sum_assignment
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
        except ImportError:
            # 如果没有scipy，使用贪婪匹配
            return BaseTracker._greedy_assignment(cost_matrix, threshold)
        
        matched_indices = []
        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] <= threshold:
                matched_indices.append((row, col))
        
        matched_rows = set(r for r, _ in matched_indices)
        matched_cols = set(c for _, c in matched_indices)
        
        unmatched_rows = [i for i in range(cost_matrix.shape[0]) if i not in matched_rows]
        unmatched_cols = [j for j in range(cost_matrix.shape[1]) if j not in matched_cols]
        
        return matched_indices, unmatched_rows, unmatched_cols
    
    @staticmethod
    def _greedy_assignment(cost_matrix: np.ndarray, threshold: float) -> Tuple[List, List, List]:
        """
        贪婪匹配（备选方案）
        
        Args:
            cost_matrix: 代价矩阵
            threshold: 匹配阈值
            
        Returns:
            匹配results
        """
        matched_indices = []
        matched_rows = set()
        matched_cols = set()
        
        if cost_matrix.size == 0:
            return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
        
        # 找到所有小于阈值的位置
        valid = np.argwhere(cost_matrix <= threshold)
        
        if len(valid) == 0:
            return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
        
        # 按代价排序
        costs = cost_matrix[valid[:, 0], valid[:, 1]]
        sorted_indices = np.argsort(costs)
        
        for idx in sorted_indices:
            row, col = valid[idx]
            if row not in matched_rows and col not in matched_cols:
                matched_indices.append((int(row), int(col)))
                matched_rows.add(row)
                matched_cols.add(col)
        
        unmatched_rows = [i for i in range(cost_matrix.shape[0]) if i not in matched_rows]
        unmatched_cols = [j for j in range(cost_matrix.shape[1]) if j not in matched_cols]
        
        return matched_indices, unmatched_rows, unmatched_cols
