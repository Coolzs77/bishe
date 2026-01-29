#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-object tracking evaluation module.

Provides MOT evaluation utilities, including MOTA, MOTP, IDF1, and related metrics.
"""

import os
import json
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from collections import defaultdict


class MOTEvaluator:
    """
    Multi-object tracking evaluator.
    
    Evaluates MOT algorithms, computing CLEAR MOT and ID metrics.
    
    Attributes:
        iou_threshold: IoU matching threshold
        frame_results: Frame-level tracking results
        frame_gts: Frame-level ground truth annotations
    """
    
    def __init__(self, iou_threshold: float = 0.5):
        """
        Initialize the MOT evaluator.
        
        Args:
            iou_threshold: IoU matching threshold
        """
        self.iou_threshold = iou_threshold
        self.reset()
    
    def reset(self) -> None:
        """重置评估器"""
        self.frame_results: Dict[int, List[Dict]] = {}
        self.frame_gts: Dict[int, List[Dict]] = {}
        
        # 累积统计量
        self.num_gt = 0
        self.num_pred = 0
        self.num_matches = 0
        self.num_false_positives = 0
        self.num_misses = 0
        self.num_switches = 0
        self.num_fragmentations = 0
        self.total_iou = 0.0
        
        # ID统计量
        self.idtp = 0
        self.idfp = 0
        self.idfn = 0
        
        # 轨迹映射
        self.gt_to_pred_mapping: Dict[int, int] = {}
        self.prev_frame_gt_ids: set = set()
        
        self.frame_count = 0
    
    def add_frame(
        self,
        frame_id: int,
        pred_boxes: np.ndarray,
        pred_ids: np.ndarray,
        gt_boxes: np.ndarray,
        gt_ids: np.ndarray
    ) -> None:
        """
        添加单帧的跟踪结果和真实标注
        
        Args:
            frame_id: 帧ID
            pred_boxes: 预测边界框，形状为 (N, 4)
            pred_ids: 预测ID，形状为 (N,)
            gt_boxes: 真实边界框，形状为 (M, 4)
            gt_ids: 真实ID，形状为 (M,)
        """
        self.frame_results[frame_id] = [
            {'bbox': box, 'id': id_} 
            for box, id_ in zip(pred_boxes, pred_ids)
        ]
        
        self.frame_gts[frame_id] = [
            {'bbox': box, 'id': id_} 
            for box, id_ in zip(gt_boxes, gt_ids)
        ]
        
        self.frame_count += 1
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        计算所有MOT评估指标
        
        Returns:
            包含各项指标的字典
        """
        # 重置累积统计量
        self.num_gt = 0
        self.num_pred = 0
        self.num_matches = 0
        self.num_false_positives = 0
        self.num_misses = 0
        self.num_switches = 0
        self.total_iou = 0.0
        self.idtp = 0
        self.idfp = 0
        self.idfn = 0
        self.gt_to_pred_mapping = {}
        self.prev_frame_gt_ids = set()
        
        # 按帧ID排序处理
        frame_ids = sorted(self.frame_results.keys())
        
        for frame_id in frame_ids:
            self._process_frame(frame_id)
        
        # 计算最终指标
        metrics = {}
        
        # MOTA (Multiple Object Tracking Accuracy)
        if self.num_gt > 0:
            mota = 1 - (self.num_misses + self.num_false_positives + self.num_switches) / self.num_gt
        else:
            mota = 0.0
        metrics['MOTA'] = mota
        
        # MOTP (Multiple Object Tracking Precision)
        if self.num_matches > 0:
            motp = self.total_iou / self.num_matches
        else:
            motp = 0.0
        metrics['MOTP'] = motp
        
        # IDF1 (ID F1 Score)
        if (2 * self.idtp + self.idfp + self.idfn) > 0:
            idf1 = 2 * self.idtp / (2 * self.idtp + self.idfp + self.idfn)
        else:
            idf1 = 0.0
        metrics['IDF1'] = idf1
        
        # 精确率
        if self.num_pred > 0:
            precision = self.num_matches / self.num_pred
        else:
            precision = 0.0
        metrics['Precision'] = precision
        
        # 召回率
        if self.num_gt > 0:
            recall = self.num_matches / self.num_gt
        else:
            recall = 0.0
        metrics['Recall'] = recall
        
        # F1分数
        if (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        metrics['F1'] = f1
        
        # 其他统计量
        metrics['num_gt'] = self.num_gt
        metrics['num_pred'] = self.num_pred
        metrics['num_matches'] = self.num_matches
        metrics['num_false_positives'] = self.num_false_positives
        metrics['num_misses'] = self.num_misses
        metrics['num_switches'] = self.num_switches
        metrics['num_frames'] = self.frame_count
        
        return metrics
    
    def _process_frame(self, frame_id: int) -> None:
        """处理单帧"""
        preds = self.frame_results.get(frame_id, [])
        gts = self.frame_gts.get(frame_id, [])
        
        num_pred = len(preds)
        num_gt = len(gts)
        
        self.num_gt += num_gt
        self.num_pred += num_pred
        
        if num_gt == 0 and num_pred == 0:
            return
        
        if num_gt == 0:
            self.num_false_positives += num_pred
            self.idfp += num_pred
            return
        
        if num_pred == 0:
            self.num_misses += num_gt
            self.idfn += num_gt
            return
        
        # 计算IoU矩阵
        pred_boxes = np.array([p['bbox'] for p in preds])
        gt_boxes = np.array([g['bbox'] for g in gts])
        
        iou_matrix = self._compute_iou_matrix(gt_boxes, pred_boxes)
        
        # 贪婪匹配
        matched_gt = set()
        matched_pred = set()
        matches = []
        
        while True:
            max_iou = self.iou_threshold
            max_gt_idx = -1
            max_pred_idx = -1
            
            for i in range(num_gt):
                if i in matched_gt:
                    continue
                for j in range(num_pred):
                    if j in matched_pred:
                        continue
                    if iou_matrix[i, j] > max_iou:
                        max_iou = iou_matrix[i, j]
                        max_gt_idx = i
                        max_pred_idx = j
            
            if max_gt_idx < 0:
                break
            
            matches.append((max_gt_idx, max_pred_idx, max_iou))
            matched_gt.add(max_gt_idx)
            matched_pred.add(max_pred_idx)
        
        # 更新统计量
        self.num_matches += len(matches)
        self.num_false_positives += num_pred - len(matches)
        self.num_misses += num_gt - len(matches)
        
        # 更新MOTP
        for gt_idx, pred_idx, iou in matches:
            self.total_iou += iou
        
        # 检查ID切换
        current_gt_ids = set()
        for gt_idx, pred_idx, _ in matches:
            gt_id = gts[gt_idx]['id']
            pred_id = preds[pred_idx]['id']
            current_gt_ids.add(gt_id)
            
            if gt_id in self.gt_to_pred_mapping:
                if self.gt_to_pred_mapping[gt_id] != pred_id:
                    self.num_switches += 1
            
            self.gt_to_pred_mapping[gt_id] = pred_id
        
        self.prev_frame_gt_ids = current_gt_ids
        
        # 更新ID统计量
        self.idtp += len(matches)
        self.idfp += num_pred - len(matches)
        self.idfn += num_gt - len(matches)
    
    @staticmethod
    def _compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """计算IoU矩阵"""
        if len(boxes1) == 0 or len(boxes2) == 0:
            return np.zeros((len(boxes1), len(boxes2)))
        
        # 扩展维度
        boxes1 = boxes1[:, np.newaxis, :]
        boxes2 = boxes2[np.newaxis, :, :]
        
        # 计算交集
        x1 = np.maximum(boxes1[..., 0], boxes2[..., 0])
        y1 = np.maximum(boxes1[..., 1], boxes2[..., 1])
        x2 = np.minimum(boxes1[..., 2], boxes2[..., 2])
        y2 = np.minimum(boxes1[..., 3], boxes2[..., 3])
        
        inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # 计算面积
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        
        # 计算IoU
        union_area = area1 + area2 - inter_area
        iou = np.where(union_area > 0, inter_area / union_area, 0)
        
        return iou.squeeze()
    
    def save_results(self, filepath: str) -> None:
        """
        保存评估结果
        
        Args:
            filepath: 保存路径
        """
        results = self.compute_metrics()
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"MOT评估结果已保存到: {filepath}")
    
    def print_results(self) -> None:
        """打印评估结果"""
        results = self.compute_metrics()
        
        print("\n" + "=" * 50)
        print("多目标跟踪评估结果")
        print("=" * 50)
        print(f"帧数: {results['num_frames']}")
        print(f"真实目标数: {results['num_gt']}")
        print(f"预测目标数: {results['num_pred']}")
        print("-" * 50)
        print(f"MOTA: {results['MOTA']:.4f}")
        print(f"MOTP: {results['MOTP']:.4f}")
        print(f"IDF1: {results['IDF1']:.4f}")
        print("-" * 50)
        print(f"精确率: {results['Precision']:.4f}")
        print(f"召回率: {results['Recall']:.4f}")
        print(f"F1: {results['F1']:.4f}")
        print("-" * 50)
        print(f"ID切换: {results['num_switches']}")
        print(f"假正例: {results['num_false_positives']}")
        print(f"漏检: {results['num_misses']}")
        print("=" * 50)


def evaluate_tracker(
    tracker,
    frames: List[np.ndarray],
    detections: List[np.ndarray],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    评估跟踪器
    
    Args:
        tracker: 跟踪器实例
        frames: 帧图像列表
        detections: 检测结果列表，每个元素形状为 (N, 4)
        ground_truths: 真实标注列表，每个元素包含 'boxes' 和 'ids'
        iou_threshold: IoU阈值
        
    Returns:
        评估结果字典
    """
    evaluator = MOTEvaluator(iou_threshold=iou_threshold)
    
    tracker.reset()
    
    for frame_id, (frame, dets, gt) in enumerate(zip(frames, detections, ground_truths)):
        # 运行跟踪器
        result = tracker.update(dets)
        
        # 获取跟踪结果
        pred_boxes = result.get_boxes()
        pred_ids = result.get_ids()
        
        # 获取真实标注
        gt_boxes = np.array(gt['boxes'])
        gt_ids = np.array(gt['ids'])
        
        # 添加到评估器
        evaluator.add_frame(frame_id, pred_boxes, pred_ids, gt_boxes, gt_ids)
    
    return evaluator.compute_metrics()


def compare_trackers(
    trackers: Dict[str, Any],
    frames: List[np.ndarray],
    detections: List[np.ndarray],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5
) -> Dict[str, Dict[str, float]]:
    """
    对比多个跟踪器
    
    Args:
        trackers: 跟踪器字典 {名称: 跟踪器实例}
        frames: 帧图像列表
        detections: 检测结果列表
        ground_truths: 真实标注列表
        iou_threshold: IoU阈值
        
    Returns:
        各跟踪器的评估结果字典
    """
    results = {}
    
    for name, tracker in trackers.items():
        print(f"评估跟踪器: {name}")
        metrics = evaluate_tracker(
            tracker, frames, detections, ground_truths, iou_threshold
        )
        results[name] = metrics
        print(f"  MOTA: {metrics['MOTA']:.4f}, IDF1: {metrics['IDF1']:.4f}")
    
    return results


def print_comparison_table(results: Dict[str, Dict[str, float]]) -> None:
    """
    打印跟踪器对比表格
    
    Args:
        results: 各跟踪器的评估结果
    """
    metrics = ['MOTA', 'MOTP', 'IDF1', 'Precision', 'Recall', 'F1', 'num_switches']
    
    print("\n" + "=" * 80)
    print("跟踪器性能对比")
    print("=" * 80)
    
    # 打印表头
    header = f"{'Tracker':<20}"
    for metric in metrics:
        header += f"{metric:<12}"
    print(header)
    print("-" * 80)
    
    # 打印各跟踪器结果
    for name, result in results.items():
        row = f"{name:<20}"
        for metric in metrics:
            value = result.get(metric, 0)
            if isinstance(value, float):
                row += f"{value:<12.4f}"
            else:
                row += f"{value:<12}"
        print(row)
    
    print("=" * 80)
