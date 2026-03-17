#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【统一指标库】
所有评估都使用这个库，确保指标一致

包含：
- 检测性能: mAP@0.5, Precision, Recall
- 跟踪性能: MOTA, IDF1, IDSW
- 系统性能: FPS, Latency, Memory, CPU
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict


class DetectionMetrics:
    """【检测性能指标】- 所有检测评估必须用这个"""

    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self):
        """重置所有统计"""
        self.tp = 0  # True Positive
        self.fp = 0  # False Positive
        self.fn = 0  # False Negative

    @staticmethod
    def calculate_iou(box1: Tuple, box2: Tuple) -> float:
        """
        计算两个框的IoU

        参数:
            box1, box2: (x1, y1, x2, y2) 格式
        返回:
            iou: [0, 1]
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0

        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area

        return inter_area / (union_area + 1e-7)

    def evaluate_frame(self, detections: List[Tuple], ground_truths: List[Tuple]):
        """
        评估单帧

        参数:
            detections: 预测框列表 [(x1, y1, x2, y2), ...]
            ground_truths: 真值框列表 [(x1, y1, x2, y2), ...]
        """
        matched_det = set()

        # 匹配每个真值框
        for gt in ground_truths:
            best_iou = 0
            best_det_idx = -1

            for det_idx, det in enumerate(detections):
                if det_idx in matched_det:
                    continue

                iou = self.calculate_iou(gt, det)
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = det_idx

            # 判断是否匹配
            if best_iou >= self.iou_threshold:
                self.tp += 1
                matched_det.add(best_det_idx)
            else:
                self.fn += 1

        # 未匹配的预测是FP
        self.fp += len(detections) - len(matched_det)

    def get_metrics(self) -> Dict:
        """
        获取检测指标

        返回:
            {
                'TP': int,
                'FP': int,
                'FN': int,
                'Precision': float (0-1),
                'Recall': float (0-1),
                'F1-Score': float (0-1),
            }
        """
        precision = self.tp / (self.tp + self.fp + 1e-7)
        recall = self.tp / (self.tp + self.fn + 1e-7)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)

        return {
            'TP': self.tp,
            'FP': self.fp,
            'FN': self.fn,
            'Precision': precision,  # [0, 1]
            'Recall': recall,  # [0, 1]
            'F1-Score': f1,  # [0, 1]
        }


class TrackingMetrics:
    """【跟踪性能指标】- 所有跟踪评估必须用这个"""

    def __init__(self):
        self.reset()

    def reset(self):
        """重置所有统计"""
        self.tracking_results = []  # [(frame_id, track_id, bbox)]
        self.ground_truths = []  # [(frame_id, obj_id, bbox)]

    def add_tracking_result(self, frame_id: int, track_id: int, bbox: Tuple):
        """添加跟踪结果"""
        self.tracking_results.append((frame_id, track_id, bbox))

    def add_ground_truth(self, frame_id: int, obj_id: int, bbox: Tuple):
        """添加真值标注"""
        self.ground_truths.append((frame_id, obj_id, bbox))

    def compute_mota(self) -> Dict:
        """
        计算MOTA (Multi-Object Tracking Accuracy)

        MOTA = 1 - (FN + FP + IDSW) / GT

        返回:
            {
                'MOTA': float (0-1),
                'FN': int,
                'FP': int,
                'IDSW': int,
                'GT': int,
            }
        """
        # 按帧组织数据
        frame_trks = defaultdict(list)
        frame_gts = defaultdict(list)

        for frame_id, track_id, bbox in self.tracking_results:
            frame_trks[frame_id].append((track_id, bbox))

        for frame_id, obj_id, bbox in self.ground_truths:
            frame_gts[frame_id].append((obj_id, bbox))

        total_fn = 0
        total_fp = 0
        total_gt = 0
        idsw = 0

        prev_matched_pairs = {}

        # 逐帧计算
        for frame_id in sorted(set(frame_gts.keys()) | set(frame_trks.keys())):
            trks = frame_trks.get(frame_id, [])
            gts = frame_gts.get(frame_id, [])

            total_gt += len(gts)

            # 匹配GT和预测
            matched_pairs = {}
            matched_trk_ids = set()

            for gt_idx, (obj_id, gt_bbox) in enumerate(gts):
                best_iou = 0
                best_trk_idx = -1

                for trk_idx, (track_id, trk_bbox) in enumerate(trks):
                    if trk_idx in matched_trk_ids:
                        continue

                    iou = DetectionMetrics.calculate_iou(gt_bbox, trk_bbox)
                    if iou > best_iou and iou >= 0.5:
                        best_iou = iou
                        best_trk_idx = trk_idx

                if best_trk_idx >= 0:
                    track_id = trks[best_trk_idx][0]
                    matched_pairs[obj_id] = track_id
                    matched_trk_ids.add(best_trk_idx)
                else:
                    total_fn += 1

            # 检测ID切换
            for obj_id, prev_track_id in prev_matched_pairs.items():
                if obj_id in matched_pairs and matched_pairs[obj_id] != prev_track_id:
                    idsw += 1

            # 未匹配的预测是FP
            total_fp += len(trks) - len(matched_trk_ids)

            prev_matched_pairs = matched_pairs

        mota = 1 - (total_fn + total_fp + idsw) / (total_gt + 1e-7)

        return {
            'MOTA': max(0, mota),  # [0, 1]
            'FN': total_fn,
            'FP': total_fp,
            'IDSW': idsw,
            'GT': total_gt,
        }

    def compute_idf1(self) -> Dict:
        """
        计算IDF1 (ID F1-score)

        IDF1 = 2*IDTP / (2*IDTP + IDFP + IDFN)

        返回:
            {
                'IDF1': float (0-1),
                'IDTP': int,
                'IDFP': int,
                'IDFN': int,
            }
        """
        # 按ID组织轨迹
        id_detections = defaultdict(list)
        id_ground_truths = defaultdict(list)

        for frame_id, track_id, bbox in self.tracking_results:
            id_detections[track_id].append((frame_id, bbox))

        for frame_id, obj_id, bbox in self.ground_truths:
            id_ground_truths[obj_id].append((frame_id, bbox))

        idtp = 0
        idfp = 0
        idfn = 0

        # 计算IDTP, IDFP
        for track_id in id_detections:
            for det_frame, det_bbox in id_detections[track_id]:
                matched = False
                for obj_id in id_ground_truths:
                    for gt_frame, gt_bbox in id_ground_truths[obj_id]:
                        if det_frame == gt_frame:
                            iou = DetectionMetrics.calculate_iou(det_bbox, gt_bbox)
                            if iou >= 0.5:
                                idtp += 1
                                matched = True
                                break
                    if matched:
                        break

                if not matched:
                    idfp += 1

        # 计算IDFN
        for obj_id in id_ground_truths:
            idfn += len(id_ground_truths[obj_id]) - len([
                (frame, bbox) for frame, bbox in id_ground_truths[obj_id]
                if any(
                    det_frame == frame and
                    DetectionMetrics.calculate_iou(bbox, det_bbox) >= 0.5
                    for track_id in id_detections
                    for det_frame, det_bbox in id_detections[track_id]
                )
            ])

        idf1 = 2 * idtp / (2 * idtp + idfp + idfn + 1e-7)

        return {
            'IDF1': idf1,  # [0, 1]
            'IDTP': idtp,
            'IDFP': idfp,
            'IDFN': idfn,
        }


class PerformanceMetrics:
    """【系统性能指标】- 所有性能评估必须用这个"""

    def __init__(self):
        self.reset()

    def reset(self):
        """重置所有统计"""
        self.frame_times = []  # 单帧处理时间列表 (秒)
        self.memory_usage = []  # 内存使用列表 (MB)
        self.cpu_usage = []  # CPU使用列表 (%)

    def record_frame(self, frame_time: float, memory_mb: float = None, cpu_percent: float = None):
        """
        记录单帧数据

        参数:
            frame_time: 处理时间 (秒)
            memory_mb: 内存使用 (MB)，可选
            cpu_percent: CPU使用 (%)，可选
        """
        self.frame_times.append(frame_time)

        if memory_mb is not None:
            self.memory_usage.append(memory_mb)

        if cpu_percent is not None:
            self.cpu_usage.append(cpu_percent)

    def get_metrics(self) -> Dict:
        """
        获取性能指标

        返回:
            {
                'FPS': float,              # 帧率
                'Avg_Latency_ms': float,  # 平均延迟(毫秒)
                'Min_Latency_ms': float,
                'Max_Latency_ms': float,
                'Avg_Memory_MB': float,    # 平均内存(MB)
                'Max_Memory_MB': float,
                'Avg_CPU_Percent': float,  # 平均CPU(%)
                'Max_CPU_Percent': float,
            }
        """
        if not self.frame_times:
            return {}

        frame_times_np = np.array(self.frame_times)

        metrics = {
            'FPS': 1.0 / np.mean(frame_times_np),
            'Avg_Latency_ms': np.mean(frame_times_np) * 1000,
            'Min_Latency_ms': np.min(frame_times_np) * 1000,
            'Max_Latency_ms': np.max(frame_times_np) * 1000,
        }

        if self.memory_usage:
            metrics['Avg_Memory_MB'] = np.mean(self.memory_usage)
            metrics['Max_Memory_MB'] = np.max(self.memory_usage)

        if self.cpu_usage:
            metrics['Avg_CPU_Percent'] = np.mean(self.cpu_usage)
            metrics['Max_CPU_Percent'] = np.max(self.cpu_usage)

        return metrics